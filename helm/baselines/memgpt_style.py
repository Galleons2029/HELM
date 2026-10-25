"""MemGPT-style paging: a bounded FIFO main context (always in the prompt) backed by an archival vector store.

Observations enter the main context; when it overflows, the oldest ones are evicted to archival storage, which the
agent can query with a recall (top-k similarity). No validity metadata, no consolidation.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

from ..backends import MemoryBackend, RecallResult, RecalledItem
from ..config import HelmConfig
from ..embedding.base import Embedder
from ..memory.claims import extract_claims
from ..memory.tokens import TokenCounter
from ..types import Observation, ReadContext
from .common import VectorLog, cap_tokens, to_recalled


class MemGPTStyleBackend(MemoryBackend):
    name = "MemGPT-style"

    def __init__(self, cfg: HelmConfig, embedder: Embedder, tokens: Optional[TokenCounter] = None, main_context: int = 8):
        super().__init__(tokens)
        self.cfg, self.embedder = cfg, embedder
        self.main_context = main_context
        self.working_steps = main_context
        self.k = cfg.memory.k1
        self.limit = cfg.memory.max_recall_tokens
        self.archive = VectorLog(embedder.dim, cfg.memory.budget_m1)
        self._queue: Deque[Observation] = deque()

    def reset_episode(self, keep_memory: bool = False) -> None:
        super().reset_episode(keep_memory)
        if not keep_memory:
            self.archive.clear()
            self._queue.clear()

    def observe(self, obs: Observation, ctx: ReadContext) -> bool:
        self._note_obs(obs)
        self._queue.append(obs)
        while len(self._queue) > self.main_context:
            old = self._queue.popleft()
            self.archive.add(old, self.embedder.encode([old.text])[0])
            self._count("WRITE")  # eviction to archival storage
        return True

    def recall(self, query: str, ctx: ReadContext) -> RecallResult:
        t0 = time.perf_counter()
        self._count("READ")
        in_ctx = [
            RecalledItem(o.text, 1, extract_claims(o.text, o.tool), False, o.status, o.step, [o.obs_id])
            for o in reversed(self._queue)
        ]
        q = self.embedder.encode([query], instruction="retrieve relevant interaction logs")[0]
        arch = [to_recalled(e, s) for e, s in self.archive.search(q, self.k)]
        items, text, used = cap_tokens(arch, self.tokens, self.limit)  # only archival text costs recall tokens
        return RecallResult(
            items=in_ctx + items, text=text, tokens=used, path="main+archival", latency_ms=(time.perf_counter() - t0) * 1e3
        )
