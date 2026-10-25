"""Long-context agent without external memory: the whole interaction history stays in the prompt (up to the window)."""

from __future__ import annotations

import time
from typing import List, Optional

from ..backends import MemoryBackend, RecallResult, RecalledItem
from ..config import HelmConfig
from ..memory.claims import extract_claims
from ..memory.tokens import TokenCounter
from ..types import Observation, ReadContext


class LongContextBackend(MemoryBackend):
    name = "Long-context"

    def __init__(self, cfg: HelmConfig, tokens: Optional[TokenCounter] = None):
        super().__init__(tokens)
        self.window = cfg.reasoner.context_window
        self._items: List[RecalledItem] = []

    def reset_episode(self, keep_memory: bool = False) -> None:
        super().reset_episode(keep_memory)
        if not keep_memory:
            self._items = []

    def observe(self, obs: Observation, ctx: ReadContext) -> bool:
        self._note_obs(obs)
        self._items.append(
            RecalledItem(obs.text, 1, extract_claims(obs.text, obs.tool), False, obs.status, obs.step, [obs.obs_id])
        )
        return True

    def prompt_history_tokens(self) -> int:
        return min(sum(self._hist_tokens), self.window)

    def recall(self, query: str, ctx: ReadContext) -> RecallResult:
        # nothing is "retrieved": the history is already in the prompt (counted via prompt_history_tokens).
        # The reasoner scans it most-recent-first.
        t0 = time.perf_counter()
        self._count("READ")
        return RecallResult(
            items=list(reversed(self._items)), text="", tokens=0, path="context", latency_ms=(time.perf_counter() - t0) * 1e3
        )
