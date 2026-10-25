"""Memory-controller interface and the text format shared by the heuristic, LLM and *learned* controllers.

The controller decides, at each step, which explicit memory operations to issue (Read / Write / Consolidate / Prune)
and produces the recall query q_t. The learned controller (Stage 1 on-policy distillation + Stage 2 GSPO, see
``helm.training``) is a causal LM that reads ``render_state(state)`` and generates ``format_ops(ops)``; the offline
controllers implement the same ``decide`` interface, so any of them can be swapped in or used as a distillation teacher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol

from ..types import Status


class OpKind(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    CONSOLIDATE = "CONSOLIDATE"
    PRUNE = "PRUNE"
    NOOP = "NOOP"


@dataclass
class MemOp:
    kind: OpKind
    query: Optional[str] = None  # recall query for READ


@dataclass
class ControllerState:
    step: int
    obs_tool: str = ""
    obs_status: Status = Status.UNKNOWN
    obs_text: str = ""
    obs_has_claims: bool = False
    version_changed: bool = False
    recall_request: Optional[str] = None  # set when the reasoner emitted <recall>...</recall>
    sizes: Dict[int, int] = field(default_factory=dict)
    budgets: Dict[int, int] = field(default_factory=dict)
    consolidations: int = 0
    max_consolidations: int = 5
    steps_since_consolidate: int = 0
    writes_this_step: int = 0


def render_state(s: ControllerState, max_obs_chars: int = 400) -> str:
    """Prompt for the controller (also the training input for OPD / GSPO)."""
    lines = [
        f"step: {s.step}",
        f"memory sizes M1/M2/M3: {s.sizes.get(1, 0)}/{s.sizes.get(2, 0)}/{s.sizes.get(3, 0)} "
        f"(budgets {s.budgets.get(1, 0)}/{s.budgets.get(2, 0)}/{s.budgets.get(3, 0)})",
        f"consolidations used: {s.consolidations}/{s.max_consolidations}; steps since last consolidate: {s.steps_since_consolidate}",
    ]
    if s.obs_text:
        lines.append(
            f"observation (tool={s.obs_tool}, status={s.obs_status.value}, has_claims={int(s.obs_has_claims)}, "
            f"version_changed={int(s.version_changed)}): {s.obs_text[:max_obs_chars]}"
        )
    if s.recall_request:
        lines.append(f"recall request: {s.recall_request}")
    lines.append("Emit memory operations, one per line, as <op>NAME</op> (READ also takes <q>query</q>).")
    return "\n".join(lines)


def format_ops(ops: List[MemOp]) -> str:
    out = []
    for op in ops:
        if op.kind == OpKind.READ:
            out.append(f"<op>READ</op><q>{(op.query or '').strip()}</q>")
        else:
            out.append(f"<op>{op.kind.value}</op>")
    return "\n".join(out) if out else "<op>NOOP</op>"


_OP_RE = re.compile(r"<op>\s*([A-Za-z]+)\s*</op>(?:\s*<q>(.*?)</q>)?", re.DOTALL)


def parse_ops(text: str) -> List[MemOp]:
    """Tolerant parser; unknown op names are skipped. Returns [NOOP] if nothing parses."""
    ops: List[MemOp] = []
    for m in _OP_RE.finditer(text):
        try:
            kind = OpKind(m.group(1).upper())
        except ValueError:
            continue
        ops.append(MemOp(kind, (m.group(2) or "").strip() or None if kind == OpKind.READ else None))
    return ops or [MemOp(OpKind.NOOP)]


class Controller(Protocol):
    name: str

    def decide(self, state: ControllerState) -> List[MemOp]: ...
