#!/usr/bin/env python3
"""Scaling protocol: horizon length, memory-store size, recall-query verbosity, and trap prevalence.

The last factor is the honest control: the synthetic gains exist only where traps exist (trap bias 0 = queries on random
observed keys). Writes one summary row per (factor, value, method).

    python scripts/run_scaling.py --episodes 20
"""
import csv

from _common import *  # noqa: F401,F403
from helm.sim import run_method, summarize

METHODS = ["flat_rag", "archrag_style", "helm"]
FACTORS = {
    "horizon": ("sim", "horizon", [64, 128, 256, 512]),
    "query_extra_tokens": ("sim", "query_extra_tokens", [0, 20, 60]),
    "trap_bias": ("sim", "query_trap_bias", [0.0, 0.35, 0.7, 1.0]),
    "budget_m1": ("cfg", "memory.budget_m1", [50, 100, 200, 400]),
}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--factors", nargs="+", default=list(FACTORS), choices=list(FACTORS))
    args = ap.parse_args()
    cfg0 = load_cfg(args)
    seeds = seeds_of(args, cfg0)
    out = []
    for fname in args.factors:
        kind, field, values = FACTORS[fname]
        for v in values:
            cfg, sim = cfg0, sim_cfg(args)
            if kind == "sim":
                setattr(sim, field, v)
            else:
                cfg = cfg0.override(**{field: v})
            cache = {}
            rows = []
            for m in METHODS:
                rows += run_method(m, cfg, sim, seeds, args.episodes, episode_cache=cache, agent_kw={"vigilant": args.vigilant})
            for s in summarize(rows):
                out.append({"factor": fname, "value": v, **{k: s[k] for k in ("method", "success", "ctu", "epierr", "fresh_per_ep")}, "source": "simulated"})
            print(f"{fname}={v}: " + "  ".join(f"{s['method']}: succ {s['success']:.1f} ctu {s['ctu']:.0f}" for s in summarize(rows)))
    path = Path(args.out_dir) / "sim_scaling_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(BANNER)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
