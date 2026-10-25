"""Epistemic governance: validity (status / freshness / consistency) and the governed score of Eq. (1).

    S(m; q) = l_sim * cos(v_q, v) + l_time * g(dt) + l_status * w(status) + l_src * w(src)

A memory item is *epistemically valid* iff status != FAILED, freshness == 1 and consistency == 1. Read applies a hard
filter (Alg. 1 subtracts infinity from invalid items) with a soft fallback: when no valid candidate exists the best
invalid ones are readmitted and flagged, so failed evidence is never silently lost (Sec. "Retaining failed traces").
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..config import GovernanceConfig
from ..types import MemoryItem, ReadContext, ScoredItem, Status
from .claims import ClaimRegistry


class Governor:
    def __init__(self, cfg: GovernanceConfig, registry: ClaimRegistry):
        self.cfg = cfg
        self.registry = registry

    # ------------------------------------------------------------------ validity
    def is_stale(self, item: MemoryItem, ctx: ReadContext) -> bool:
        g = self.cfg
        if item.meta.get("structural"):
            return False
        if ctx.step - item.step_id > g.step_window:
            return True
        if ctx.time > 0 and item.timestamp > 0 and (ctx.time - item.timestamp) > g.time_window_days * 86400.0:
            return True
        if g.check_version and ctx.task_version is not None and item.task_version is not None:
            if item.task_version != ctx.task_version:
                return True
        return False

    def invalid_reasons(self, item: MemoryItem, ctx: ReadContext) -> List[str]:
        """Subset of {"failed", "stale", "contradicted"} that applies to ``item`` under the enabled signals."""
        g = self.cfg
        if not g.enabled or item.meta.get("structural"):
            return []
        reasons: List[str] = []
        exempt = g.diagnostic_exemption and ctx.diagnostic
        if g.use_status and item.status == Status.FAILED and not exempt:
            reasons.append("failed")
        if g.use_freshness and self.is_stale(item, ctx):
            reasons.append("stale")
        if g.use_consistency and self.registry.is_contradicted(item.claims, ctx.step):
            reasons.append("contradicted")
        return reasons

    # ------------------------------------------------------------------ score
    def recency(self, item: MemoryItem, ctx: ReadContext) -> float:
        dt = max(0, ctx.step - item.step_id)
        return math.exp(-dt / max(self.cfg.recency_scale, 1e-6))

    def score(self, item: MemoryItem, sim: float, ctx: ReadContext) -> float:
        g = self.cfg
        s = g.lambda_sim * sim
        if not g.enabled:
            return s
        if g.use_freshness and not item.meta.get("structural"):
            s += g.lambda_time * self.recency(item, ctx)
        if g.use_status:
            lam = 0.0 if (g.diagnostic_exemption and ctx.diagnostic) else g.lambda_status
            s += lam * g.status_weights.get(item.status.value, 0.0)
        if g.use_source:
            s += g.lambda_src * g.source_weights.get(item.source.value, 0.0)
        return s

    # ------------------------------------------------------------------ filter + rank
    def rank(
        self, cands: Sequence[Tuple[MemoryItem, float]], ctx: ReadContext, k: int
    ) -> Tuple[List[ScoredItem], List[ScoredItem], bool]:
        """Return ``(selected, dropped, fallback_used)`` for ``cands = [(item, cosine_sim)]``."""
        g = self.cfg
        scored = []
        for item, sim in cands:
            reasons = self.invalid_reasons(item, ctx)
            scored.append(ScoredItem(item, self.score(item, sim, ctx), sim, valid=not reasons, reasons=reasons))
        scored.sort(key=lambda s: s.score, reverse=True)
        if not (g.enabled and g.hard_filter):
            # Soft mode: no validity judgement is surfaced, items are only re-ranked by S (the role of lambda_status).
            for s in scored:
                s.valid = True
            return scored[:k], [], False
        valid = [s for s in scored if s.valid]
        dropped = [s for s in scored if not s.valid]
        if valid:
            return valid[:k], dropped, False
        if g.fallback_readmit and scored:
            readmit = scored[: min(k, g.fallback_k)]
            for s in readmit:
                s.readmitted = True
            return readmit, [], True
        return [], dropped, False
