#!/usr/bin/env python3
"""Component, per-signal, layer and traversal ablations of HELM on the synthetic benchmark.

    python scripts/run_ablations.py --episodes 30
"""
from _common import *  # noqa: F401,F403
from helm.methods import ABLATIONS
from helm.sim import run_grid


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg, sim = load_cfg(args), sim_cfg(args)
    seeds = seeds_of(args, cfg)
    rows = run_grid(["flat_rag", "helm_lite", "helm", *ABLATIONS], cfg, sim, seeds, args.episodes, agent_kw={"vigilant": args.vigilant})
    finish("sim_ablations", rows, args)


if __name__ == "__main__":
    main()
