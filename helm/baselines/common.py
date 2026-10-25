"""Shared pieces for the baseline backends. Every baseline uses the same embedder, the same per-step recall-token cap
and the same entry budget as HELM (paper: matched budgets), so differences come from the memory design alone.

NOTE: these are lightweight *in-harness approximations* of each paradigm (flat episodic RAG, long context, MemGPT-style
paging, Mem0-style fact upsert, ArchRAG-style hierarchy without governance). They are not the original systems;
HippoRAG, GraphRAG, MemGen and G-Memory are intentionally not re-implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..backends import RecalledItem
from ..index import FlatIndex
from ..memory.claims import extract_claims
from ..types import Observation, Status


@dataclass
class Entry:
    uid: int
    text: str
    vec: np.ndarray
    tool: str
    obs_id: str
    step: int
    status: Status
    claims: List[Tuple[str, str]] = field(default_factory=list)


class VectorLog:
    """Capacity-bounded (FIFO) similarity store."""

    def __init__(self, dim: int, capacity: int):
        self.capacity = capacity
        self.index = FlatIndex(dim)
        self.entries: Dict[int, Entry] = {}
        self._next = 0

    def clear(self) -> None:
        self.__init__(self.index.dim, self.capacity)  # type: ignore[misc]

    def add(self, obs: Observation, vec: np.ndarray, claims: Optional[List[Tuple[str, str]]] = None) -> Entry:
        e = Entry(self._next, obs.text, vec, obs.tool, obs.obs_id, obs.step, obs.status,
                  claims if claims is not None else extract_claims(obs.text, obs.tool))
        self.entries[e.uid] = e
        self.index.add(str(e.uid), vec)
        self._next += 1
        while len(self.entries) > self.capacity:
            oldest = min(self.entries)
            self.index.remove(str(oldest))
            del self.entries[oldest]
        return e

    def search(self, q: np.ndarray, k: int) -> List[Tuple[Entry, float]]:
        return [(self.entries[int(i)], s) for i, s in self.index.search(q, k) if int(i) in self.entries]

    def __len__(self) -> int:
        return len(self.entries)


def to_recalled(e: Entry, score: float = 0.0) -> RecalledItem:
    return RecalledItem(
        text=e.text, layer=1, claims=e.claims, flagged_invalid=False, status=e.status,
        step_id=e.step, audit_ids=[e.obs_id], score=score,
    )


def cap_tokens(items: List[RecalledItem], tokens, limit: int) -> Tuple[List[RecalledItem], str, int]:
    """Keep items in order until the recall-token cap is hit."""
    kept, lines, used = [], [], 0
    for it in items:
        c = tokens.count(it.text)
        if used + c > limit:
            break
        kept.append(it)
        lines.append(it.text)
        used += c
    return kept, "\n".join(lines), used
