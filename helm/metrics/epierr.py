"""Epistemic-error accounting (Appendix "Metric sheet").

EpiErr counts consumed evidence that is *semantically relevant but epistemically invalid*: stale (version mismatch or
age beyond the window), failed (tool status), or contradictory (a newer SUCCESS claim with the same key and a different
value). Everything is judged from logged observation metadata kept in an ``AuditLog`` that is independent of the memory
backend under test -- never from task outcomes -- so the metric cannot be gamed by the system it evaluates and is not
circular with governance (a backend's own validity flags are not consulted).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import GovernanceConfig
from ..memory.claims import extract_claims
from ..types import Observation, ReadContext, Status


class AuditLog:
    def __init__(self, cfg: Optional[GovernanceConfig] = None, unknown_counts_as_failed: bool = False):
        self.cfg = cfg or GovernanceConfig()
        # The main text defines validity as status != FAILED; the metric-sheet wording says "non-success". We default to
        # the former (consistent with Sec. 3.2) and expose the latter as a switch.
        self.unknown_counts_as_failed = unknown_counts_as_failed
        self.obs: Dict[str, Observation] = {}
        self._claims: Dict[str, List[Tuple[int, str, Status, str]]] = defaultdict(list)

    def reset(self) -> None:
        self.obs.clear()
        self._claims.clear()

    def add(self, obs: Observation) -> None:
        self.obs[obs.obs_id] = obs
        for k, v in extract_claims(obs.text, obs.tool):
            self._claims[k].append((obs.step, v, obs.status, obs.obs_id))

    def classify(
        self, audit_ids: List[str], key: str, value: str, ctx: ReadContext, truth: Optional[str] = None
    ) -> List[str]:
        """Invalidity reasons of the evidence an item provides for ``key == value`` at ``ctx``.

        Staleness has two modes. With ``truth=None`` it follows the paper's metadata rule (version mismatch or age beyond
        the window) -- the same rule governance filters on, so governed systems score ~0 by construction. With ``truth``
        (the world's current value, available only in simulation) an item is stale iff its value no longer matches the
        world, which is independent of what governance can see.
        """
        if not audit_ids or audit_ids[0] not in self.obs:
            return []
        o = self.obs[audit_ids[0]]
        g = self.cfg
        reasons: List[str] = []
        if o.status == Status.FAILED or (self.unknown_counts_as_failed and o.status != Status.SUCCESS):
            reasons.append("failed")
        if truth is not None:
            stale = value != truth
        else:
            stale = ctx.step - o.step > g.step_window
            if ctx.time > 0 and o.timestamp > 0 and ctx.time - o.timestamp > g.time_window_days * 86400.0:
                stale = True
            if g.check_version and ctx.task_version is not None and o.task_version is not None:
                stale |= o.task_version != ctx.task_version
        if stale:
            reasons.append("stale")
        for step, v, st, oid in self._claims.get(key, ()):
            if st == Status.SUCCESS and o.step < step <= ctx.step and v != value:
                reasons.append("contradicted")
                break
        return reasons


@dataclass
class EpiErrMeter:
    consumed: int = 0
    invalid: int = 0
    by_reason: Dict[str, int] = field(default_factory=lambda: {"failed": 0, "stale": 0, "contradicted": 0})

    def add(self, reasons: List[str]) -> None:
        self.consumed += 1
        if reasons:
            self.invalid += 1
            for r in set(reasons):
                self.by_reason[r] = self.by_reason.get(r, 0) + 1

    @property
    def rate(self) -> float:
        """EpiErr as a percentage of consumed recalls."""
        return 100.0 * self.invalid / self.consumed if self.consumed else float("nan")
