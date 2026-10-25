"""Effective context tokens (CTU).

    CTU = sum_t Tok( system || history_t || retrieved_t || tool_obs_t )

summed over the steps of an episode ("total input-context tokens consumed", Appendix "Metric sheet").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CTUMeter:
    system_tokens: int = 40  # fixed system prompt / tool schema size per call (ASSUMPTION; identical for all methods)
    total: int = 0
    per_step: List[int] = field(default_factory=list)

    def step(self, history_tokens: int, retrieved_tokens: int, tool_obs_tokens: int) -> int:
        t = self.system_tokens + history_tokens + retrieved_tokens + tool_obs_tokens
        self.total += t
        self.per_step.append(t)
        return t

    @property
    def mean_per_step(self) -> float:
        return self.total / len(self.per_step) if self.per_step else 0.0
