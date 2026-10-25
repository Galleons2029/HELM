"""Claim extraction and contradiction tracking (Appendix "Metric sheet": ExtractClaims).

``extract_claims`` uses tool-type-specific parsers for structured outputs (JSON / key-value) and a constrained regex
for ``key=value`` assertions in free text, followed by simple normalisation. Two claims *contradict* when they share a
key and disagree on value; the newer SUCCESS-supported claim wins.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from ..types import Claim, Status

# Fields that describe the tool call rather than assert a fact about the world.
_META_FIELDS = {"code", "status", "version", "ver", "ts", "timestamp", "error", "msg", "message", "step", "latency_ms"}
_SUBJECT_FIELDS = ("entity", "subject", "id", "name")
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
# Constrained free-text extractor: key=value (no spaces around '=' inside machine-ish text) or "key = value".
_KV = re.compile(r"(?P<k>[A-Za-z_][\w\.\-]{1,60})\s*=\s*(?P<v>-?\d[\d,]*\.?\d*|[A-Za-z][\w\-\./]*)")
_NOISE_KEYS = {"code", "step", "ver", "v", "tool", "log", "field", "start", "end", "status", "version"}

Parser = Callable[[str], List[Tuple[str, str]]]
_PARSERS: Dict[str, Parser] = {}


def register_parser(tool_prefix: str, fn: Parser) -> None:
    """Register a tool-type-specific parser; matched by tool-name prefix, longest prefix first."""
    _PARSERS[tool_prefix] = fn


def normalize_value(v: object) -> str:
    s = str(v).strip().strip("\"'").replace(",", "").lower()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(round(f, 6))
    except ValueError:
        return s


def _json_claims(text: str) -> List[Tuple[str, str]]:
    m = _JSON_OBJ.search(text)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(d, dict):
        return []
    subject = next((str(d[f]) for f in _SUBJECT_FIELDS if f in d and not isinstance(d[f], (dict, list))), None)
    out: List[Tuple[str, str]] = []
    if subject and "attr" in d and "value" in d:  # canonical (entity, attr, value) record
        return [(f"{subject}.{d['attr']}".lower(), normalize_value(d["value"]))]
    if "key" in d and "value" in d:
        return [(str(d["key"]).lower(), normalize_value(d["value"]))]
    for k, v in d.items():
        if k in _META_FIELDS or k in _SUBJECT_FIELDS or isinstance(v, (dict, list)):
            continue
        key = f"{subject}.{k}" if subject else str(k)
        out.append((key.lower(), normalize_value(v)))
    return out


def _regex_claims(text: str) -> List[Tuple[str, str]]:
    out = []
    for m in _KV.finditer(text):
        k = m.group("k").lower()
        if k in _NOISE_KEYS:
            continue
        out.append((k, normalize_value(m.group("v"))))
    return out


def extract_claims(text: str, tool: str = "") -> List[Tuple[str, str]]:
    """Return ``[(key, normalised_value)]``. Tool-specific parser first, then JSON, then constrained regex."""
    for prefix in sorted(_PARSERS, key=len, reverse=True):
        if tool.startswith(prefix):
            got = _PARSERS[prefix](text)
            if got:
                return got
    got = _json_claims(text)
    if got:
        return got
    return _regex_claims(text)


class ClaimRegistry:
    """Key -> claims seen so far; answers "is this claim contradicted by a newer verified claim?"."""

    def __init__(self) -> None:
        self._by_key: Dict[str, List[Claim]] = defaultdict(list)

    def add(self, claim: Claim) -> None:
        self._by_key[claim.key].append(claim)

    def discard_item(self, item_id: str) -> None:
        for k, lst in self._by_key.items():
            self._by_key[k] = [c for c in lst if c.item_id != item_id]

    def newest_verified(self, key: str, before_step: Optional[int] = None) -> Optional[Claim]:
        best = None
        for c in self._by_key.get(key, ()):
            if c.status != Status.SUCCESS:
                continue
            if before_step is not None and c.step_id > before_step:
                continue
            if best is None or c.step_id >= best.step_id:
                best = c
        return best

    def contradicting(self, claim: Claim, upto_step: Optional[int] = None) -> Optional[Claim]:
        """A newer SUCCESS claim with the same key and a different value, if any."""
        newest = self.newest_verified(claim.key, upto_step)
        if newest is None or newest.item_id == claim.item_id:
            return None
        if newest.step_id > claim.step_id and newest.value != claim.value:
            return newest
        return None

    def is_contradicted(self, claims: List[Claim], upto_step: Optional[int] = None) -> bool:
        return any(self.contradicting(c, upto_step) is not None for c in claims)

    def keys(self) -> List[str]:
        return list(self._by_key)
