#!/usr/bin/env python3
"""Main comparison on the synthetic epistemic-trap benchmark: baselines vs HELM, incl. the 2x2 retriever comparison.

    python scripts/run_sim.py --episodes 30            # seeds 0 42 123, all methods
    python scripts/run_sim.py --vigilant               # reasoner that distrusts raw error codes (stronger baselines)
"""
from _common import *  # noqa: F401,F403
from helm.sim import run_grid

METHODS = [
    "long_context", "flat_rag", "flat_rag+emb", "memgpt_style", "mem0_style",
    "archrag_style", "archrag_style+emb", "helm_lite", "helm_no_emb", "helm",
]


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()
    cfg, sim = load_cfg(args), sim_cfg(args)
    seeds = seeds_of(args, cfg)
    print(f"seeds={list(seeds)} episodes/seed={args.episodes} vigilant={args.vigilant} sim={sim}")
    rows = run_grid(args.methods, cfg, sim, seeds, args.episodes, agent_kw={"vigilant": args.vigilant})
    finish("sim_main" + ("_vigilant" if args.vigilant else ""), rows, args)


if __name__ == "__main__":
    main()
