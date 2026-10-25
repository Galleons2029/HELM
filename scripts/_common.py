"""Shared CLI plumbing for the simulator scripts (path setup, config loading, output writing)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helm.config import HelmConfig  # noqa: E402
from helm.sim import SimConfig, format_table, summarize, write_rows, write_summary  # noqa: E402

BANNER = (
    "NOTE: synthetic epistemic-trap benchmark. Every row is labelled source=simulated; these numbers say how a memory\n"
    "mechanism behaves on a controlled trap structure and are NOT comparable to the paper's benchmark results."
)


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--overlay", nargs="*", default=[], help="extra YAML files layered on --config")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default: cfg.seeds (0 42 123)")
    ap.add_argument("--episodes", type=int, default=50, help="episodes per seed")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--vigilant", action="store_true", help="reasoner distrusts raw error codes even without annotations")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--trap-bias", type=float, default=None)
    return ap


def load_cfg(args) -> HelmConfig:
    return HelmConfig.from_yaml(args.config, *args.overlay)


def sim_cfg(args, **over) -> SimConfig:
    s = SimConfig()
    if getattr(args, "horizon", None):
        s.horizon = args.horizon
    if getattr(args, "trap_bias", None) is not None:
        s.query_trap_bias = args.trap_bias
    for k, v in over.items():
        setattr(s, k, v)
    return s


def seeds_of(args, cfg: HelmConfig) -> Sequence[int]:
    return tuple(args.seeds) if args.seeds else tuple(cfg.seeds)


def finish(name: str, rows, args, cols=("success", "ctu", "epierr", "epierr_meta", "fresh_per_ep"), extra: Dict | None = None) -> None:
    out = Path(args.out_dir)
    write_rows(rows, out / f"{name}_episodes.csv")
    summ = summarize(rows)
    write_summary(summ, out / f"{name}_summary.csv")
    (out / f"{name}_meta.json").write_text(
        json.dumps({"source": "simulated", "episodes_per_seed": args.episodes, "vigilant": args.vigilant, **(extra or {})}, indent=2)
    )
    print(BANNER)
    print()
    print(format_table(summ, cols))
    print(f"\nwrote {out / (name + '_episodes.csv')} and {name}_summary.csv")
