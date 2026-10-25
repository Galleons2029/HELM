"""Run memory methods on the synthetic benchmark and aggregate the results.

Every row written carries ``source = "simulated"``: these numbers come from a controlled trap-structured simulator and are
never comparable with the paper's benchmark results.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ..config import HelmConfig
from ..methods import build_method
from ..metrics.stats import bootstrap_ci, mean_se
from .agent import EpisodeMetrics, SimAgent
from .world import Episode, SimConfig

SOURCE = "simulated"


@dataclass
class Row:
    method: str
    seed: int
    episode: int
    success: float
    ctu: int
    epierr: float
    epierr_meta: float
    n_queries: int
    n_fresh: int
    n_recall_miss: int
    n_interventions: int
    n_fallback: int
    epi_stale: int
    epi_failed: int
    epi_contradicted: int
    recall_tokens: int
    latency_ms: float
    cite_cov: float = float("nan")
    cite_valid: float = float("nan")
    cite_prec: float = float("nan")
    source: str = SOURCE


def _row(name: str, seed: int, ep_idx: int, m: EpisodeMetrics) -> Row:
    return Row(name, seed, ep_idx, m.success, m.ctu, m.epierr, m.epierr_meta, m.n_queries, m.n_fresh, m.n_recall_miss,
               m.n_interventions, m.n_fallback, m.epi_stale, m.epi_failed, m.epi_contradicted, m.n_recall_tokens,
               m.recall_latency_ms, m.cite_cov, m.cite_valid, m.cite_prec)


def run_method(
    method: str,
    cfg: HelmConfig,
    sim: SimConfig,
    seeds: Sequence[int],
    n_episodes: int,
    label: Optional[str] = None,
    episode_cache: Optional[Dict] = None,
    agent_kw: Optional[Dict] = None,
    **build_kw,
) -> List[Row]:
    """One method over ``len(seeds) * n_episodes`` identical episodes (same episodes for every method)."""
    rows: List[Row] = []
    for seed in seeds:
        be = build_method(method, cfg, seed=seed, **build_kw)
        agent = SimAgent(be, cfg, **(agent_kw or {}))
        for e in range(n_episodes):
            key = (seed, e, repr(sim))
            if episode_cache is not None and key in episode_cache:
                ep = episode_cache[key]
            else:
                ep = Episode(sim, seed * 100_000 + e)
                if episode_cache is not None:
                    episode_cache[key] = ep
            rows.append(_row(label or be.name, seed, e, agent.run(ep)))
    return rows


def run_grid(
    methods: Iterable[str],
    cfg: HelmConfig,
    sim: SimConfig,
    seeds: Sequence[int] = (0, 42, 123),
    n_episodes: int = 30,
    log: Optional[Callable[[str], None]] = print,
    agent_kw: Optional[Dict] = None,
) -> List[Row]:
    cache: Dict = {}
    rows: List[Row] = []
    for m in methods:
        t0 = time.time()
        r = run_method(m, cfg, sim, seeds, n_episodes, episode_cache=cache, agent_kw=agent_kw)
        rows.extend(r)
        if log:
            log(f"  {m:<22s} {len(r):4d} episodes  {time.time() - t0:5.1f}s")
    return rows


# ---------------------------------------------------------------------- aggregation / io
def write_rows(rows: Sequence[Row], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def summarize(rows: Sequence[Row]) -> List[Dict[str, object]]:
    """Per method: mean over seeds (+SE) of seed-level means, plus a bootstrap CI over episodes for success."""
    out = []
    for method in dict.fromkeys(r.method for r in rows):
        mr = [r for r in rows if r.method == method]
        seeds = sorted({r.seed for r in mr})
        per_seed = lambda f: [sum(f(r) for r in mr if r.seed == s) / max(1, sum(1 for r in mr if r.seed == s)) for s in seeds]
        succ, succ_se = mean_se(per_seed(lambda r: 100 * r.success))
        ctu, ctu_se = mean_se(per_seed(lambda r: r.ctu))
        # EpiErr pooled per seed over consumed recalls would need counts; per-episode mean is used (NaN-safe).
        epi_vals = [[r.epierr for r in mr if r.seed == s and not math.isnan(r.epierr)] for s in seeds]
        epi, epi_se = mean_se([sum(v) / len(v) for v in epi_vals if v])
        meta_vals = [[r.epierr_meta for r in mr if r.seed == s and not math.isnan(r.epierr_meta)] for s in seeds]
        epim, epim_se = mean_se([sum(v) / len(v) for v in meta_vals if v])
        _, lo, hi = bootstrap_ci([100 * r.success for r in mr])
        n = max(1, len(mr))
        cite = lambda f: (lambda v: sum(v) / len(v) if v else float("nan"))([f(r) for r in mr if not math.isnan(f(r))])
        out.append({
            "method": method, "n_episodes": len(mr), "success": succ, "success_se": succ_se,
            "success_ci_lo": lo, "success_ci_hi": hi, "ctu": ctu, "ctu_se": ctu_se,
            "epierr": epi, "epierr_se": epi_se, "epierr_meta": epim, "epierr_meta_se": epim_se,
            "fresh_per_ep": sum(r.n_fresh for r in mr) / n,
            "interventions_per_ep": sum(r.n_interventions for r in mr) / n,
            "recall_tokens_per_ep": sum(r.recall_tokens for r in mr) / n,
            "cite_cov": cite(lambda r: r.cite_cov), "cite_valid": cite(lambda r: r.cite_valid),
            "cite_prec": cite(lambda r: r.cite_prec),
            "source": SOURCE,
        })
    return out


def format_table(summary: Sequence[Dict[str, object]], cols: Sequence[str] = ("success", "ctu", "epierr", "fresh_per_ep")) -> str:
    head = ["method", *cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for s in summary:
        cells = [str(s["method"])]
        for c in cols:
            v = s[c]
            se = s.get(f"{c}_se")
            if isinstance(v, float) and math.isnan(v):
                cells.append("N/A")
            else:
                cells.append(f"{v:.1f} ± {se:.1f}" if se is not None and c in ("success", "ctu", "epierr", "epierr_meta") else f"{v:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(summary: Sequence[Dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
