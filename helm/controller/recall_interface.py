"""Event-driven recall interface: ``<recall>...</recall>`` -> governed retrieval -> ``<memory>...</memory>``.

The reasoner emits a recall request span; the *memory generator* runs governed retrieval, optionally expands
provenance, and replaces the span with an auditable ``<memory>`` block. Three generator variants (Appendix, faithfulness
controls):

    A0  default recall: the governed retrieval text as-is.
    A1  hard cited recall: every sentence carries an inline ``[[CITE ...]]``; the block is validated and regenerated
        (up to R times, diagnostics appended) and falls back to A0 / extractive-only recall on persistent failure.
    A2  extractive-first: select provenance spans (with citations) first, optionally compress afterwards.

Offline, generation is deterministic (cite the provenance pointer of each recalled item). With an LLM client, A1
prompts the model with the fixed instruction from the paper and relies on ``cited_recall_loop`` for validation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..llm.client import ChatClient
from ..memory.provenance import cited_recall_loop, validate_cited_memory
from ..types import RetrievalResult

RECALL_RE = re.compile(r"<recall>(.*?)</recall>", re.DOTALL)

GENERATOR_PROMPT = (
    "You write the <memory> block for an agent. Use ONLY the evidence spans provided. Do not make unsupported claims. "
    "Do not cite spans that are not listed. Every sentence inside <memory> must end with at least one citation of the "
    "form [[CITE tool=<t> log=<log_id> step=<s> field=<f> start=<i> end=<j>]]."
)


def extract_recall_requests(text: str) -> List[str]:
    return [m.strip() for m in RECALL_RE.findall(text)]


def replace_recalls(text: str, memory_blocks: List[str]) -> str:
    it = iter(memory_blocks)
    return RECALL_RE.sub(lambda m: next(it, ""), text)


def _evidence_lines(store, result: RetrievalResult) -> List[Tuple[str, str]]:
    """(sentence, cite) for each recalled item that carries a provenance pointer."""
    out = []
    for s in result.items:
        if not (s.valid or s.readmitted) or not s.item.provenance:
            continue
        out.append((s.item.text.rstrip(". "), s.item.provenance[0].as_cite()))
    return out


class MemoryGenerator:
    def __init__(self, variant: str = "A0", client: Optional[ChatClient] = None, max_retries: int = 3):
        assert variant in ("A0", "A1", "A2")
        self.variant = variant
        self.client = client
        self.max_retries = max_retries
        self.last_info: Dict[str, object] = {}

    def generate(self, store, request: str, result: RetrievalResult) -> str:
        if self.variant == "A0":
            self.last_info = {"variant": "A0"}
            return f"<memory>{result.text}</memory>"
        lines = _evidence_lines(store, result)
        if self.variant == "A2":
            return self._extractive(store, result, lines)
        return self._cited(store, request, result, lines)

    # ------------------------------------------------------------------ A1
    def _cited(self, store, request: str, result: RetrievalResult, lines: List[Tuple[str, str]]) -> str:
        def det_generate(_req: str) -> str:
            body = " ".join(f"{s} {c}." for s, c in lines)  # cite inside the sentence, period after
            return f"<memory>{body}</memory>"

        def llm_generate(req: str) -> str:
            ev = "\n".join(f"- {s} {c}" for s, c in lines)
            msgs = [
                {"role": "system", "content": GENERATOR_PROMPT},
                {"role": "user", "content": f"Request: {req}\nEvidence:\n{ev}"},
            ]
            return self.client.chat(msgs)  # type: ignore[union-attr]

        gen = llm_generate if self.client is not None else det_generate
        if not lines:
            self.last_info = {"variant": "A1", "fallback": True, "attempts": 0, "codes": []}
            return f"<memory>{result.text}</memory>"
        out, info = cited_recall_loop(
            gen, request, store.spans, self.max_retries, fallback=lambda _r: f"<memory>{result.text}</memory>"
        )
        self.last_info = {"variant": "A1", **info}
        return out

    # ------------------------------------------------------------------ A2
    def _extractive(self, store, result: RetrievalResult, lines: List[Tuple[str, str]]) -> str:
        parts = []
        for s in result.items:
            if not (s.valid or s.readmitted):
                continue
            for p in s.item.provenance[:1]:
                span = store.spans.resolve(p)
                if span:
                    parts.append(f"{span} {p.as_cite()}")
        if not parts:
            self.last_info = {"variant": "A2", "fallback": True}
            return f"<memory>{result.text}</memory>"
        self.last_info = {"variant": "A2", "fallback": False}
        # sentence-ify so the validator sees one citation per sentence
        return "<memory>" + " ".join(p.rstrip(".") + "." if not p.endswith("]]") else p for p in parts) + "</memory>"

    def validate(self, store, y: str) -> Tuple[bool, str, str]:
        return validate_cited_memory(y, store.spans)
