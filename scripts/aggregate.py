#!/usr/bin/env python3
"""Collect every ``outputs/*_summary.csv`` into one Markdown report (+ a CTU-vs-success scatter if matplotlib exists).

    python scripts/aggregate.py

The report states, per table, where the numbers came from (``source`` column). Nothing here reads the paper's
``results/`` directory and nothing is written there.
"""
import csv
import math
import os
from pathlib import Path

from _common import ROOT

OUT = ROOT / "outputs"
COLS = ("success", "ctu", "epierr", "epierr_meta", "fresh_per_ep", "cite_cov", "cite_valid", "cite_prec")


def read(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(v: str) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v or ""
    return "N/A" if math.isnan(x) else (f"{x:.0f}" if abs(x) >= 1000 else f"{x:.1f}")


def table(rows, cols):
    head = ["method", *cols]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join([r["method"], *[fmt(r.get(c, "")) for c in cols]]) + " |")
    return "\n".join(out)


def main() -> None:
    parts = [
        "# Synthetic-benchmark report\n",
        "All rows below have `source = simulated`. They come from a controlled trap-structured simulator and say nothing "
        "about GAIA / WebArena / Mind2Web / StreamBench / LifelongAgentBench performance.\n",
    ]
    for path in sorted(OUT.glob("sim_*_summary.csv")):
        rows = read(path)
        if not rows:
            continue
        if "factor" in rows[0]:
            parts += [f"## {path.stem}\n", "| factor | value | method | success | ctu | epierr |", "|---|---|---|---|---|---|"]
            parts += [f"| {r['factor']} | {r['value']} | {r['method']} | {fmt(r['success'])} | {fmt(r['ctu'])} | {fmt(r['epierr'])} |" for r in rows]
            parts.append("")
            continue
        cols = [c for c in COLS if c in rows[0] and any(fmt(r[c]) != "N/A" for r in rows)]
        parts += [f"## {path.stem}\n", table(rows, cols), ""]
    report = OUT / "REPORT.md"
    report.write_text("\n".join(parts))
    print(f"wrote {report}")
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        main_csv = OUT / "sim_main_summary.csv"
        if main_csv.exists():
            rows = read(main_csv)
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in rows:
                ax.scatter(float(r["ctu"]), float(r["success"]), s=28)
                ax.annotate(r["method"], (float(r["ctu"]), float(r["success"])), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel("CTU per episode (log) -- simulated")
            ax.set_ylabel("task success % -- simulated")
            ax.set_title("Synthetic epistemic-trap benchmark (NOT paper data)", fontsize=8)
            fig.tight_layout()
            fig.savefig(OUT / "sim_success_vs_ctu.png", dpi=160)
            print(f"wrote {OUT / 'sim_success_vs_ctu.png'}")
    except ImportError:
        print("matplotlib not available: skipped the scatter plot")


if __name__ == "__main__":
    main()
