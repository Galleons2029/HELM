#!/usr/bin/env python3
"""Synthesize hierarchical triplets (M1/M2/M3), cross-layer queries, long-horizon haystacks and epistemic counterfactuals.

    python scripts/synthesize_data.py --smoke                                  # offline, tiny, writes outputs/synth_smoke/
    python scripts/synthesize_data.py --n 2000 --out outputs/synth/triplets.jsonl   # offline template generator
    python scripts/synthesize_data.py --generator teacher --seeds-file seeds.txt \\
        --teacher-model Qwen/Qwen3-8B --base-url http://localhost:8000/v1       # NOT verified: needs a served teacher LLM
    python scripts/synthesize_data.py --generator teacher --fineweb-edu 500 ... # NOT verified: needs `datasets` + network

The offline generator is verified (tests/test_training_math.py). The teacher generator is verified only against ``MockLLM``;
live endpoints and the FineWeb-Edu loader are not covered by the offline test suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.training import print_config, write_metrics  # noqa: E402
from helm.training.data_synthesis import (  # noqa: E402
    SynthConfig,
    dataset_stats,
    generate_offline_dataset,
    generate_teacher_dataset,
    load_fineweb_edu_seeds,
    write_jsonl,
)


def read_seed_texts(path: str) -> list[str]:
    """One seed per line; a JSON line with a ``text`` field is also accepted."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            line = str(json.loads(line).get("text", ""))
        if line:
            out.append(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny offline run (seconds); stats are labelled source=smoke")
    ap.add_argument("--generator", choices=("offline", "teacher"), default="offline")
    ap.add_argument("--n", type=int, default=1000, help="number of records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output JSONL (default outputs/synth[_smoke]/triplets.jsonl)")
    ap.add_argument("--long-frac", type=float, default=0.25, help="fraction of needle-in-a-haystack records")
    ap.add_argument("--min-ctx-tokens", type=int, default=512)
    ap.add_argument("--max-ctx-tokens", type=int, default=8192)
    ap.add_argument("--n-counterfactuals", type=int, default=3)
    ap.add_argument("--seeds-file", default=None, help="teacher generator: text file with one seed per line")
    ap.add_argument("--fineweb-edu", type=int, default=0, help="teacher generator: draw this many FineWeb-Edu seeds")
    ap.add_argument("--min-edu-score", type=float, default=3.0)
    ap.add_argument("--teacher-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--teacher-temperature", type=float, default=0.7)
    args = ap.parse_args(argv)

    if args.smoke:
        args.generator, args.n, args.long_frac = "offline", 64, 0.2
        args.min_ctx_tokens, args.max_ctx_tokens = 160, 300  # scaled down so smoke records stay short
    out = Path(args.out or ("outputs/synth_smoke/triplets.jsonl" if args.smoke else "outputs/synth/triplets.jsonl"))
    cfg = SynthConfig(n=args.n, seed=args.seed, n_counterfactuals=args.n_counterfactuals, long_frac=args.long_frac,
                      min_ctx_tokens=args.min_ctx_tokens, max_ctx_tokens=args.max_ctx_tokens)
    print_config("synthesize_data", {"args": vars(args), "synth_config": cfg.__dict__, "output": str(out)})

    n_failed = 0
    if args.generator == "offline":
        records = generate_offline_dataset(cfg)
    else:
        from helm.llm.client import OpenAICompatClient

        if args.fineweb_edu:
            seeds = load_fineweb_edu_seeds(args.fineweb_edu, min_score=args.min_edu_score)
        elif args.seeds_file:
            seeds = read_seed_texts(args.seeds_file)
        else:
            ap.error("--generator teacher needs --seeds-file or --fineweb-edu")
        client = OpenAICompatClient(model=args.teacher_model, base_url=args.base_url, api_key_env=args.api_key_env)
        records, n_failed = generate_teacher_dataset(client, seeds, cfg, temperature=args.teacher_temperature)

    n = write_jsonl(out, records)
    stats = dataset_stats(records)
    stats["teacher_failures"] = n_failed
    metrics = write_metrics(out.with_name("stats.json"), "smoke" if args.smoke else "measured",
                            generator=args.generator, config=cfg.__dict__, stats=stats)
    print(f"wrote {n} records -> {out}\nwrote stats -> {metrics}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
