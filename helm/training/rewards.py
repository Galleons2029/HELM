"""Hybrid reward for Stage 2 (GSPO): ``R = alpha * r_task + beta * r_mem``.

Paper-stated: the hybrid form and ``alpha = 1.0``, ``beta = 0.05``; r_task is the benchmark task metric; r_mem rewards low
effective context tokens and penalises redundant writes and epistemically risky memory access (e.g. promoting
stale/failed traces).

ASSUMPTIONS (the paper gives no formula for r_mem):
  * ``memory_reward`` = ``(1 - min(ctu_tokens / ctu_budget, 1)) - w_redundant * (n_redundant_writes / n_ops)
    - w_risky * (n_risky_reads / n_ops)`` with ``w_redundant = w_risky = 0.5``, so it is bounded in [-1, 1].
  * The toy task reward below (``toy_task_reward``) is NOT a benchmark metric. It scores a controller completion
    against the ``HeuristicController`` decision and exists only so the GSPO smoke test has a verifiable reward.
    In the toy setting the CTU proxy is the completion length in tokens, a "redundant write" is any WRITE beyond the
    first (paper: write <= 1 per step), and a "risky read" is a READ whose query differs from the recall request.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from ..config import ControllerConfig
from ..controller.base import ControllerState, MemOp, OpKind, parse_ops
from ..controller.heuristic import HeuristicController
from .data_synthesis import state_from_dict

_DEFAULTS = ControllerConfig()  # alpha = 1.0, beta = 0.05 (paper)


def hybrid_reward(r_task: float, r_mem: float, alpha: float = _DEFAULTS.alpha, beta: float = _DEFAULTS.beta) -> float:
    """``R = alpha * r_task + beta * r_mem``."""
    return alpha * r_task + beta * r_mem


def memory_reward(
    ctu_tokens: float,
    ctu_budget: float,
    n_redundant_writes: int,
    n_risky_reads: int,
    n_ops: int,
    w_redundant: float = 0.5,
    w_risky: float = 0.5,
) -> float:
    """Bounded memory-cost reward in [-1, 1] (formula is an ASSUMPTION, see module docstring)."""
    if ctu_budget <= 0:
        raise ValueError("ctu_budget must be positive")
    # ASSUMPTION: the whole formula below (the paper only lists the three ingredients, not how they combine).
    token_term = 1.0 - min(max(ctu_tokens, 0.0) / ctu_budget, 1.0)
    denom = max(n_ops, 1)
    red = min(max(n_redundant_writes, 0) / denom, 1.0)
    risky = min(max(n_risky_reads, 0) / denom, 1.0)
    return max(-1.0, min(1.0, token_term - w_redundant * red - w_risky * risky))


# ------------------------------------------------------------------------------------------------ toy (smoke) reward
def _norm(q: Optional[str]) -> str:
    return " ".join((q or "").lower().split())


def required_ops(state: ControllerState) -> List[MemOp]:
    """Ops the offline teacher policy takes in ``state``."""
    return HeuristicController().decide(state)


def _has_op_tags(text: str) -> bool:
    return "<op>" in text


def toy_task_reward(state: ControllerState, completion: str) -> float:
    """In [0, 1]: Jaccard overlap of op kinds with the heuristic decision, halved if a required READ misses its query."""
    if not _has_op_tags(completion):
        return 0.0
    want = required_ops(state)
    got = parse_ops(completion)
    kinds_w, kinds_g = {o.kind for o in want}, {o.kind for o in got}
    jaccard = len(kinds_w & kinds_g) / len(kinds_w | kinds_g)
    if state.recall_request:
        read_ok = any(o.kind == OpKind.READ and _norm(o.query) == _norm(state.recall_request) for o in got)
    else:
        read_ok = True
    return jaccard if read_ok else 0.5 * jaccard


def toy_memory_stats(state: ControllerState, completion: str) -> Dict[str, int]:
    got = parse_ops(completion) if _has_op_tags(completion) else []
    kinds = Counter(o.kind for o in got)
    return {
        "n_ops": len(got),
        "n_redundant_writes": max(kinds[OpKind.WRITE] - 1, 0),
        "n_risky_reads": sum(
            1 for o in got if o.kind == OpKind.READ and _norm(o.query) != _norm(state.recall_request)
        ),
    }


def toy_hybrid_reward(
    state: ControllerState,
    completion: str,
    n_tokens: int,
    ctu_budget: float = 64.0,  # ASSUMPTION: toy token budget
    alpha: float = _DEFAULTS.alpha,
    beta: float = _DEFAULTS.beta,
) -> Dict[str, float]:
    r_task = toy_task_reward(state, completion)
    st = toy_memory_stats(state, completion)
    r_mem = memory_reward(n_tokens, ctu_budget, st["n_redundant_writes"], st["n_risky_reads"], st["n_ops"])
    return {"reward": hybrid_reward(r_task, r_mem, alpha, beta), "r_task": r_task, "r_mem": r_mem}


RewardFn = Callable[[Dict[str, Any], str, int], Dict[str, float]]


def make_toy_reward_fn(alpha: float = _DEFAULTS.alpha, beta: float = _DEFAULTS.beta, ctu_budget: float = 64.0) -> RewardFn:
    """Reward callable for ``GSPOTrainer``: ``fn(record, completion_text, n_completion_tokens) -> {reward, r_task, r_mem}``.

    ``record`` is a ``make_synthetic_controller_dataset`` record (it carries the serialised ``ControllerState``).
    """

    def fn(record: Dict[str, Any], completion: str, n_tokens: int) -> Dict[str, float]:
        if "state" not in record:
            raise KeyError("the toy reward needs a serialised `state` in each record; use --reward-fn for other data")
        return toy_hybrid_reward(state_from_dict(record["state"]), completion, n_tokens, ctu_budget, alpha, beta)

    return fn
