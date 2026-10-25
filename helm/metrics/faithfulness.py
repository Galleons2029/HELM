"""Provenance faithfulness and citation metrics.

Provenance faithfulness = fraction of recalled sentences that can be matched to at least one valid provenance pointer
whose span contains supporting evidence (Appendix "Metric sheet"). In the cited-recall variant this reduces to citation
coverage / validity / precision. The paper judges C-P with a fixed LLM judge on 100 GAIA-L2 recalls; here the judge is a
pluggable ``support_fn`` and the default is a transparent lexical-overlap proxy (not the paper's judge).
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional

from ..memory.provenance import CITE_RE, Citation, SpanTable, _SENT_SPLIT, _memory_block, citation_metrics

_STOP = {"as", "of", "step", "the", "a", "an", "is", "at", "version", "to", "in", "and"}
_TOK = re.compile(r"[a-z0-9]+")


def lexical_support(sentence: str, span: str, threshold: float = 0.6) -> bool:
    s = [t for t in _TOK.findall(sentence.lower()) if t not in _STOP]
    if not s:
        return False
    sp = set(_TOK.findall(span.lower()))
    return sum(t in sp for t in s) / len(s) >= threshold


def provenance_faithfulness(
    memory_block: str, spans: SpanTable, support_fn: Optional[Callable[[str, str], bool]] = None
) -> Dict[str, float]:
    """Coverage / validity / precision plus the paper-level faithfulness (supported sentences / sentences)."""
    support_fn = support_fn or lexical_support
    m = citation_metrics(memory_block, spans, support_fn)
    block = _memory_block(memory_block) or ""
    sents = [s.strip() for s in _SENT_SPLIT.split(block) if s.strip()]
    ok = 0
    for s in sents:
        for cm in CITE_RE.finditer(s):
            c = Citation.parse(cm.group("body"))
            if c is None:
                continue
            raw = spans._fields.get((c.tool, c.log_id, c.step_id, c.field))
            if raw is not None and 0 <= c.start <= c.end <= len(raw) and support_fn(CITE_RE.sub("", s).strip(), raw[c.start : c.end]):
                ok += 1
                break
    m["faithfulness"] = ok / len(sents) if sents else float("nan")
    return m
