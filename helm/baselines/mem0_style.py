"""Mem0-style memory: extract atomic facts from each observation and ADD / UPDATE / NOOP them in a key-value fact store.

The update rule is "newer value overwrites older" (the offline stand-in for Mem0's LLM-judged update). It resolves
simple contradictions by recency but has no notion of tool status or environment version, so a failed trace that carries
a plausible value can overwrite a verified one -- exactly the failure mode HELM's governance targets.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from ..backends import MemoryBackend, RecallResult, RecalledItem
from ..config import HelmConfig
from ..embedding.base import Embedder
from ..index import FlatIndex
from ..memory.claims import extract_claims
from ..memory.tokens import TokenCounter
from ..types import Observation, ReadContext, Status
from .common import cap_tokens


class Mem0StyleBackend(MemoryBackend):
    name = "Mem0-style"

    def __init__(self, cfg: HelmConfig, embedder: Embedder, tokens: Optional[TokenCounter] = None):
        super().__init__(tokens)
        self.cfg, self.embedder = cfg, embedder
        self.k = cfg.memory.k1
        self.limit = cfg.memory.max_recall_tokens
        self.capacity = cfg.memory.budget_m2 * 2  # fact store is smaller than a raw-log store
        self._reset()

    def _reset(self) -> None:
        self.facts: Dict[str, Tuple[str, int, str, Status]] = {}  # key -> (value, step, obs_id, status)
        self.index = FlatIndex(self.embedder.dim)

    def reset_episode(self, keep_memory: bool = False) -> None:
        super().reset_episode(keep_memory)
        if not keep_memory:
            self._reset()

    def observe(self, obs: Observation, ctx: ReadContext) -> bool:
        self._note_obs(obs)
        wrote = False
        for key, value in extract_claims(obs.text, obs.tool):
            cur = self.facts.get(key)
            if cur is not None and cur[0] == value:
                self._count("NOOP")
                continue
            self._count("UPDATE" if cur is not None else "ADD")
            self.facts[key] = (value, obs.step, obs.obs_id, obs.status)
            self.index.add(key, self.embedder.encode([f"{key} = {value}"])[0])
            wrote = True
            if len(self.facts) > self.capacity:
                oldest = min(self.facts, key=lambda k: self.facts[k][1])
                self.index.remove(oldest)
                del self.facts[oldest]
        return wrote

    def recall(self, query: str, ctx: ReadContext) -> RecallResult:
        t0 = time.perf_counter()
        self._count("READ")
        q = self.embedder.encode([query], instruction="retrieve relevant facts")[0]
        items = []
        for key, sim in self.index.search(q, self.k):
            value, step, obs_id, status = self.facts[key]
            items.append(
                RecalledItem(f"{key} = {value}", 2, [(key, value)], False, status, step, [obs_id], sim)
            )
        items, text, used = cap_tokens(items, self.tokens, self.limit)
        return RecallResult(items=items, text=text, tokens=used, path="facts", latency_ms=(time.perf_counter() - t0) * 1e3)
