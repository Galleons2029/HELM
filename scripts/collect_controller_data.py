#!/usr/bin/env python3
"""Write simulator-derived controller training pairs (prompt / completion JSONL) to outputs/.

    python scripts/collect_controller_data.py --episodes 40 --seed 0
"""
import argparse

from _common import ROOT  # noqa: F401
from helm.sim.controller_data import collect_controller_dataset, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "controller_sim_pairs.jsonl"))
    args = ap.parse_args()
    rows = collect_controller_dataset(args.episodes, args.seed)
    write_jsonl(rows, args.out)
    kinds = {}
    for r in rows:
        for line in r["completion"].splitlines():
            k = line.split("</op>")[0].replace("<op>", "")
            kinds[k] = kinds.get(k, 0) + 1
    print(f"wrote {len(rows)} pairs to {args.out}  (op counts: {kinds})  source=simulated")


if __name__ == "__main__":
    main()
