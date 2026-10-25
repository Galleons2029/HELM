"""Provenance pointers, the append-only raw-log span table, and cited-recall validation.

Pointer schema (Appendix "Provenance and citation schema"): ``(tool, log_id, step_id, field, start, end, hash)``.
Inline citation: ``[[CITE tool=<t> log=<log_id> step=<s> field=<f> start=<i> end=<j>]]``.
``validate_cited_memory`` / ``cited_recall_loop`` implement the two appendix algorithms (hard cited recall with
regeneration and fallback).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from ..types import ProvPointer


def span_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class SpanTable:
    """Append-only store of raw log fields, the thing pointers resolve against.

    This is the "raw tool log" the paper's pointers refer to. It is *not* the budgeted M1 retrieval index: pruning M1
    entries never invalidates a pointer, which is what keeps recalls auditable after compression.
    """

    def __init__(self) -> None:
        self._fields: Dict[Tuple[str, str, int, str], str] = {}

    def add_log(self, tool: str, log_id: str, step_id: int, fields: Dict[str, str]) -> None:
        for name, text in fields.items():
            self._fields[(tool, log_id, step_id, name)] = text

    def pointer(
        self, tool: str, log_id: str, step_id: int, field: str = "text", start: int = 0, end: Optional[int] = None
    ) -> ProvPointer:
        text = self._fields[(tool, log_id, step_id, field)]
        end = len(text) if end is None else end
        return ProvPointer(tool, log_id, step_id, field, start, end, span_hash(text[start:end]))

    def resolve(self, p: ProvPointer, check_hash: bool = True) -> Optional[str]:
        text = self._fields.get((p.tool, p.log_id, p.step_id, p.field))
        if text is None or not (0 <= p.start <= p.end <= len(text)):
            return None
        span = text[p.start : p.end]
        if check_hash and span_hash(span) != p.hash:
            return None
        return span

    def __len__(self) -> int:
        return len(self._fields)


# ----------------------------------------------------------------------------- citations
CITE_RE = re.compile(r"\[\[CITE\s+(?P<body>[^\]]*)\]\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Citation:
    tool: str
    log_id: str
    step_id: int
    field: str
    start: int
    end: int

    @classmethod
    def parse(cls, body: str) -> Optional["Citation"]:
        kv = dict(re.findall(r"(\w+)=(\S+)", body))
        try:
            return cls(kv["tool"], kv["log"], int(kv["step"]), kv["field"], int(kv["start"]), int(kv["end"]))
        except (KeyError, ValueError):
            return None


def _memory_block(y: str) -> Optional[str]:
    m = re.search(r"<memory>(.*?)</memory>", y, re.DOTALL)
    return m.group(1).strip() if m else None


def validate_cited_memory(y: str, spans: SpanTable, pointers: Optional[Dict[Tuple, str]] = None) -> Tuple[bool, str, str]:
    """Algorithm ``ValidateCitedMemory``. Returns ``(valid, code, diagnostic)``.

    Error codes: MissingMemoryBlock, MissingCite, MalformedCite, UnresolvedPointer, HashMismatch, OK.
    A citation carries no hash inline; hash validity is checked against the table entry for the cited span, so a
    pointer whose span lies outside the log (or whose log is unknown) is ``UnresolvedPointer`` and a pointer that
    resolves but whose registered hash disagrees is ``HashMismatch``. ``pointers`` optionally maps
    ``(tool, log, step, field, start, end) -> expected_hash`` for the citations the generator was allowed to emit.
    """
    block = _memory_block(y)
    if block is None:
        return False, "MissingMemoryBlock", "output has no <memory>...</memory> block"
    sentences = [s.strip() for s in _SENT_SPLIT.split(block) if s.strip()]
    for s in sentences:
        cites = list(CITE_RE.finditer(s))
        if not cites:
            return False, "MissingCite", f"sentence has no [[CITE ...]]: {s[:80]!r}"
        for m in cites:
            c = Citation.parse(m.group("body"))
            if c is None:
                return False, "MalformedCite", f"cannot parse citation: {m.group(0)!r}"
            raw = spans._fields.get((c.tool, c.log_id, c.step_id, c.field))
            if raw is None or not (0 <= c.start <= c.end <= len(raw)):
                return False, "UnresolvedPointer", f"citation does not resolve: {m.group(0)!r}"
            if pointers is not None:
                key = (c.tool, c.log_id, c.step_id, c.field, c.start, c.end)
                expected = pointers.get(key)
                if expected is not None and span_hash(raw[c.start : c.end]) != expected:
                    return False, "HashMismatch", f"span hash differs: {m.group(0)!r}"
    return True, "OK", ""


def cited_recall_loop(
    generate: Callable[[str], str],
    request: str,
    spans: SpanTable,
    max_retries: int,
    fallback: Callable[[str], str],
    pointers: Optional[Dict[Tuple, str]] = None,
) -> Tuple[str, Dict[str, object]]:
    """Algorithm "Hard cited recall with regeneration and fallback". Returns ``(output, info)``."""
    info: Dict[str, object] = {"attempts": 0, "fallback": False, "codes": []}
    r = request
    for _ in range(max_retries):
        info["attempts"] = int(info["attempts"]) + 1  # type: ignore[arg-type]
        y = generate(r)
        ok, code, diag = validate_cited_memory(y, spans, pointers)
        info["codes"].append(code)  # type: ignore[union-attr]
        if ok:
            return y, info
        r = request + "\n[validator] " + code + ": " + diag
    info["fallback"] = True
    return fallback(request), info


# ----------------------------------------------------------------------------- citation metrics
def citation_metrics(y: str, spans: SpanTable, support_fn: Optional[Callable[[str, str], bool]] = None) -> Dict[str, float]:
    """Coverage (sentences with >=1 citation), validity (citations that resolve), precision (judged supported).

    ``support_fn(sentence, cited_span_text)`` stands in for the fixed judge model used for C-P in the paper; when it is
    None, precision is NaN rather than silently reported.
    """
    block = _memory_block(y) or ""
    sentences = [s.strip() for s in _SENT_SPLIT.split(block) if s.strip()]
    if not sentences:
        return {"coverage": float("nan"), "valid": float("nan"), "precision": float("nan")}
    covered = n_cites = n_valid = n_supported = 0
    for s in sentences:
        cites = [Citation.parse(m.group("body")) for m in CITE_RE.finditer(s)]
        if cites:
            covered += 1
        for c in cites:
            n_cites += 1
            if c is None:
                continue
            raw = spans._fields.get((c.tool, c.log_id, c.step_id, c.field))
            if raw is not None and 0 <= c.start <= c.end <= len(raw):
                n_valid += 1
                if support_fn is not None and support_fn(CITE_RE.sub("", s).strip(), raw[c.start : c.end]):
                    n_supported += 1
    return {
        "coverage": covered / len(sentences),
        "valid": n_valid / n_cites if n_cites else float("nan"),
        "precision": (n_supported / n_cites if n_cites else float("nan")) if support_fn is not None else float("nan"),
    }
