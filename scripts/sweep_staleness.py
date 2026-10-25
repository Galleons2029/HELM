#!/usr/bin/env python3
"""Sensitivity to the staleness window (+-50% around the paper's 200 steps) and to the version check.

    python scripts/sweep_staleness.py
"""
from _common import *  # noqa: F401,F403
from helm.sim import run_method


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--windows", nargs="+", type=int, default=[50, 100, 200, 300])
    args = ap.parse_args()
    cfg, sim = load_cfg(args), sim_cfg(args)
    seeds = seeds_of(args, cfg)
    rows, cache = [], {}
    for w in args.windows:
        c = cfg.override(**{"governance.step_window": w})
        rows += run_method("helm", c, sim, seeds, args.episodes, label=f"step_window={w}", episode_cache=cache, agent_kw={"vigilant": args.vigilant})
    c = cfg.override(**{"governance.check_version": False})
    rows += run_method("helm", c, sim, seeds, args.episodes, label="version_check=off", episode_cache=cache, agent_kw={"vigilant": args.vigilant})
    finish("sim_staleness", rows, args)


if __name__ == "__main__":
    main()
