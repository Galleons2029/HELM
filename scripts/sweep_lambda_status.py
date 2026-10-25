#!/usr/bin/env python3
"""Sensitivity of HELM to the status-penalty weight lambda_status (paper default 10.0), plus the diagnostic exemption.

    python scripts/sweep_lambda_status.py --values 0 1 10 100
"""
from _common import *  # noqa: F401,F403
from helm.sim import run_method


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--values", nargs="+", type=float, default=[0.0, 1.0, 10.0, 100.0])
    ap.add_argument("--modes", nargs="+", choices=["hard", "soft"], default=["hard", "soft"],
                    help="hard: validity filter on (lambda only reorders valid items); soft: filter off, so lambda_status "
                         "alone separates failed from successful traces (the paper's 'read-time ranking' role)")
    args = ap.parse_args()
    cfg, sim = load_cfg(args), sim_cfg(args)
    seeds = seeds_of(args, cfg)
    rows, cache = [], {}
    for mode in args.modes:
        for v in args.values:
            c = cfg.override(**{"governance.lambda_status": v, "governance.hard_filter": mode == "hard"})
            rows += run_method("helm", c, sim, seeds, args.episodes, label=f"{mode} filter, lambda_status={v:g}",
                               episode_cache=cache, agent_kw={"vigilant": args.vigilant})
    finish("sim_lambda_status", rows, args, extra={"lambda_status_values": args.values})


if __name__ == "__main__":
    main()
