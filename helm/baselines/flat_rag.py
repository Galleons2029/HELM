"""Flat episodic memory (RAG): every observation is stored, recall is top-k cosine similarity, nothing else."""

from __future__ import annotations

import time
from typing import Optional

from ..backends import MemoryBackend, RecallResult
from ..config import HelmConfig
from ..embedding.base import Embedder
from ..memory.tokens import TokenCounter
from ..types import Observation, ReadContext
from .common import VectorLog, cap_tokens, to_recalled


class FlatRAGBackend(MemoryBackend):
    name = "Flat RAG"

    def __init__(self, cfg: HelmConfig, embedder: Embedder, tokens: Optional[TokenCounter] = None):
        super().__init__(tokens)
        self.cfg, self.embedder = cfg, embedder
        self.k = cfg.memory.k1
        self.limit = cfg.memory.max_recall_tokens
        self.log = VectorLog(embedder.dim, cfg.memory.budget_m1)

    def reset_episode(self, keep_memory: bool = False) -> None:
        super().reset_episode(keep_memory)
        if not keep_memory:
            self.log.clear()

    def observe(self, obs: Observation, ctx: ReadContext) -> bool:
        self._note_obs(obs)
        self.log.add(obs, self.embedder.encode([obs.text])[0])
        self._count("WRITE")
        return True

    def recall(self, query: str, ctx: ReadContext) -> RecallResult:
        t0 = time.perf_counter()
        self._count("READ")
        q = self.embedder.encode([query], instruction="retrieve relevant interaction logs")[0]
        hits = self.log.search(q, self.k)
        items, text, used = cap_tokens([to_recalled(e, s) for e, s in hits], self.tokens, self.limit)
        return RecallResult(items=items, text=text, tokens=used, path="M1", latency_ms=(time.perf_counter() - t0) * 1e3)
