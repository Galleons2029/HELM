"""Uniform memory-backend interface so HELM, its ablations and every baseline run under the *same* harness, budgets and
token accounting (paper: "all methods share the same agent scaffold and tool interfaces; we vary only the memory
system and its controller").
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import HelmConfig
from .controller.base import Controller, ControllerState, OpKind
from .controller.heuristic import HeuristicController, RuleController
from .controller.recall_interface import MemoryGenerator
from .embedding.base import Embedder
from .memory.claims import extract_claims
from .memory.shnm import SHNM
from .memory.tokens import DEFAULT_COUNTER, TokenCounter
from .types import Observation, ReadContext, Status


@dataclass
class RecalledItem:
    """One recalled item as the reasoner sees it, plus the metadata evaluation needs (never shown to the reasoner)."""

    text: str
    layer: int = 1
    claims: List[Tuple[str, str]] = field(default_factory=list)  # (key, value) pairs asserted by the item
    flagged_invalid: bool = False  # the memory block marks this item invalid (only governed backends do)
    status: Optional[Status] = None
    step_id: int = -1
    audit_ids: List[str] = field(default_factory=list)  # observation ids it derives from (for EpiErr)
    score: float = 0.0


@dataclass
class RecallResult:
    items: List[RecalledItem] = field(default_factory=list)
    text: str = ""
    tokens: int = 0  # tokens injected into the prompt by this recall
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    path: str = ""
    latency_ms: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


class MemoryBackend(ABC):
    name = "backend"
    annotated = False  # does the recalled block expose epistemic metadata to the reasoner?
    working_steps = 3  # raw recent observations kept verbatim in the prompt

    def __init__(self, tokens: Optional[TokenCounter] = None):
        self.tokens = tokens or DEFAULT_COUNTER
        self.op_counts: Dict[str, int] = {}
        self._hist_tokens: List[int] = []

    # --- lifecycle -------------------------------------------------------
    def reset_episode(self, keep_memory: bool = False) -> None:
        if not keep_memory:  # continual settings keep the raw history window together with the memory
            self._hist_tokens = []

    def _count(self, op: str, n: int = 1) -> None:
        self.op_counts[op] = self.op_counts.get(op, 0) + n

    def _note_obs(self, obs: Observation) -> None:
        self._hist_tokens.append(self.tokens.count(obs.text))

    def prompt_history_tokens(self) -> int:
        """Tokens of raw history the reasoner carries in its prompt at this step (excluding retrieved memory)."""
        return sum(self._hist_tokens[-self.working_steps :])

    # --- interface -------------------------------------------------------
    @abstractmethod
    def observe(self, obs: Observation, ctx: ReadContext) -> bool:
        """Show the backend a new observation; returns True if it was written to memory."""

    @abstractmethod
    def recall(self, query: str, ctx: ReadContext) -> RecallResult: ...

    def stats(self) -> Dict[str, Any]:
        return dict(self.op_counts)


def _item_claims(text: str, tool: str = "") -> List[Tuple[str, str]]:
    return extract_claims(text, tool)


class HelmBackend(MemoryBackend):
    """SHNM + epistemic governance + a controller (+ memory-generator variant). Covers HELM, HELM-lite, ArchRAG-style
    and every ablation: they differ only in ``HelmConfig`` and controller."""

    def __init__(
        self,
        cfg: HelmConfig,
        embedder: Embedder,
        controller: Optional[Controller] = None,
        generator: Optional[MemoryGenerator] = None,
        name: str = "HELM",
        seed: int = 0,
        tokens: Optional[TokenCounter] = None,
    ):
        super().__init__(tokens)
        self.name = name
        self.cfg = cfg
        self.store = SHNM(cfg, embedder, self.tokens, seed=seed)
        self.annotated = cfg.governance.enabled
        if controller is None:
            controller = RuleController() if cfg.controller.kind == "rule" else HeuristicController(cfg.memory.consolidate_every)
        self.controller = controller
        self.generator = generator or MemoryGenerator("A0")
        self._last_version: Optional[str] = None
        self._since_consolidate = 0

    def reset_episode(self, keep_memory: bool = False) -> None:
        super().reset_episode(keep_memory)
        self.store.new_episode(keep_memory=keep_memory)
        self._last_version = None
        self._since_consolidate = 0

    def _state(self, step: int, obs: Optional[Observation], recall: Optional[str]) -> ControllerState:
        m = self.cfg.memory
        return ControllerState(
            step=step,
            obs_tool=obs.tool if obs else "",
            obs_status=obs.status if obs else Status.UNKNOWN,
            obs_text=obs.text if obs else "",
            obs_has_claims=bool(obs and extract_claims(obs.text, obs.tool)),
            version_changed=bool(obs and self._last_version is not None and obs.task_version != self._last_version),
            recall_request=recall,
            sizes=self.store.sizes(),
            budgets={1: m.budget_m1, 2: m.budget_m2, 3: m.budget_m3},
            consolidations=self.store._consolidations,
            max_consolidations=m.max_consolidations_per_episode,
            steps_since_consolidate=self._since_consolidate,
        )

    def observe(self, obs: Observation, ctx: ReadContext) -> bool:
        self._note_obs(obs)
        ops = self.controller.decide(self._state(obs.step, obs, None))
        wrote = False
        for op in ops:
            self._count(op.kind.value)
            if op.kind == OpKind.WRITE:
                wrote = self.store.write(obs) is not None
            elif op.kind == OpKind.CONSOLIDATE:
                if self.store.consolidate(ctx):
                    self._since_consolidate = 0
            elif op.kind == OpKind.PRUNE:
                self.store.prune(ctx)
        self._since_consolidate += 1
        self._last_version = obs.task_version
        return wrote

    def recall(self, query: str, ctx: ReadContext) -> RecallResult:
        t0 = time.perf_counter()
        ops = self.controller.decide(self._state(ctx.step, None, query))
        q = next((o.query for o in ops if o.kind == OpKind.READ and o.query), query)
        self._count("READ")
        res = self.store.read(q, ctx)
        block = self.generator.generate(self.store, q, res)
        text = block[len("<memory>") : -len("</memory>")] if block.startswith("<memory>") else block
        items = []
        for s in res.items:
            it = s.item
            items.append(
                RecalledItem(
                    text=self.store.render(it, s.reasons if (not s.valid or s.readmitted) else None, s.readmitted),
                    layer=it.layer,
                    claims=[(c.key, c.value) for c in it.claims] or _item_claims(it.text, it.tool),
                    flagged_invalid=(not s.valid) or s.readmitted,
                    status=it.status,
                    step_id=it.step_id,
                    audit_ids=list(it.audit_ids),
                    score=s.score,
                )
            )
        tokens = self.tokens.count(text) if text else 0
        return RecallResult(
            items=items,
            text=text,
            tokens=tokens,
            interventions=res.interventions,
            fallback_used=res.fallback_used,
            path=res.path,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            info=dict(self.generator.last_info),
        )

    def stats(self) -> Dict[str, Any]:
        d = dict(self.op_counts)
        d["sizes"] = self.store.sizes()
        return d
