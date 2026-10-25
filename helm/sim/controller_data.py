"""Collect (controller state -> teacher ops) pairs from simulator episodes.

The output is the record format used by the controller-training pipeline: ``prompt = render_state(state)``,
``completion = format_ops(ops)`` (offline heuristic policy as teacher) and ``state`` (what the toy GSPO reward needs). It grounds Stage-1 distillation smoke runs
in states a real memory store actually visits (sizes, budgets, version changes, recall requests) instead of random ones.
THESE ARE SYNTHETIC STATES, not benchmark trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..config import HelmConfig
from ..controller.base import Controller, ControllerState, MemOp, format_ops, render_state
from ..controller.heuristic import HeuristicController
from ..methods import build_method
from .agent import SimAgent
from .world import Episode, SimConfig


class RecordingController:
    name = "recording"

    def __init__(self, inner: Controller):
        self.inner = inner
        self.records: List[tuple] = []

    def decide(self, state: ControllerState) -> List[MemOp]:
        ops = self.inner.decide(state)
        self.records.append((state, ops))
        return ops


def collect_controller_dataset(
    n_episodes: int = 20, seed: int = 0, cfg: Optional[HelmConfig] = None, sim: Optional[SimConfig] = None,
    teacher: Optional[Controller] = None,
) -> List[Dict[str, str]]:
    cfg, sim = cfg or HelmConfig(), sim or SimConfig()
    rec = RecordingController(teacher or HeuristicController())
    be = build_method("helm", cfg, seed=seed, controller=rec)
    agent = SimAgent(be, cfg)
    for e in range(n_episodes):
        agent.run(Episode(sim, seed * 100_000 + e))
    from ..training.data_synthesis import state_to_dict  # torch-free; keeps the record schema identical

    return [
        {"prompt": render_state(s), "completion": format_ops(o), "state": state_to_dict(s), "source": "simulated"}
        for s, o in rec.records
    ]


def write_jsonl(rows: List[Dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
