"""Offline controllers.

``RuleController``       "Rule-based" row of the controller ablation: write every observation, consolidate on a fixed
                         schedule, read whenever asked (paper: write rate 1.00).
``HeuristicController``  a hand-written policy with the same action space as the learned controller. It skips
                         low-information observations, consolidates when evidence has accumulated or memory is nearly
                         full, prunes after a version change, and rewrites nothing. It is the default offline controller
                         and the *teacher* used for offline on-policy-distillation smoke runs.
"""

from __future__ import annotations

from typing import List

from .base import ControllerState, MemOp, OpKind


class RuleController:
    name = "rule"

    def __init__(self, consolidate_every: int = 20):
        self.consolidate_every = consolidate_every

    def decide(self, s: ControllerState) -> List[MemOp]:
        ops: List[MemOp] = []
        if s.recall_request:
            ops.append(MemOp(OpKind.READ, s.recall_request))
        if s.obs_text:
            ops.append(MemOp(OpKind.WRITE))
        if s.steps_since_consolidate >= self.consolidate_every and s.consolidations < s.max_consolidations:
            ops.append(MemOp(OpKind.CONSOLIDATE))
        if any(s.sizes.get(l, 0) >= s.budgets.get(l, 1 << 30) for l in (1, 2, 3)):
            ops.append(MemOp(OpKind.PRUNE))
        return ops or [MemOp(OpKind.NOOP)]


class HeuristicController:
    name = "heuristic"

    def __init__(self, consolidate_every: int = 16, fullness: float = 0.9):
        self.consolidate_every = consolidate_every
        self.fullness = fullness

    def decide(self, s: ControllerState) -> List[MemOp]:
        ops: List[MemOp] = []
        if s.recall_request:
            ops.append(MemOp(OpKind.READ, s.recall_request))
        if s.obs_text:
            # failed traces are kept (with status) as verifiable evidence; plain noise (no claim, successful) is not
            if s.obs_has_claims or s.obs_status.value == "FAILED":
                ops.append(MemOp(OpKind.WRITE))
        near_full = s.sizes.get(1, 0) >= self.fullness * s.budgets.get(1, 1 << 30)
        due = s.steps_since_consolidate >= self.consolidate_every
        if (due or near_full) and s.consolidations < s.max_consolidations and s.sizes.get(1, 0) > 0:
            ops.append(MemOp(OpKind.CONSOLIDATE))
        if s.version_changed or any(s.sizes.get(l, 0) >= s.budgets.get(l, 1 << 30) for l in (1, 2, 3)):
            ops.append(MemOp(OpKind.PRUNE))
        return ops or [MemOp(OpKind.NOOP)]
