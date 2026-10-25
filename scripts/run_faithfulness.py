#!/usr/bin/env python3
"""Faithfulness controls A0 (default recall) / A1 (hard cited recall) / A2 (extractive-first) on the synthetic benchmark.

Citation precision uses a lexical-overlap proxy judge, NOT the paper's fixed LLM judge.
"""
from _common import *  # noqa: F401,F403
from helm.methods import FAITHFULNESS
from helm.sim import run_grid


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg, sim = load_cfg(args), sim_cfg(args)
    rows = run_grid(FAITHFULNESS, cfg, sim, seeds_of(args, cfg), args.episodes, agent_kw={"vigilant": args.vigilant})
    finish("sim_faithfulness", rows, args, cols=("success", "ctu", "epierr", "cite_cov", "cite_valid", "cite_prec"))


if __name__ == "__main__":
    main()
