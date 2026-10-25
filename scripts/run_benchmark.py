#!/usr/bin/env python3
"""Run one memory method on one real-benchmark adapter and append the results to CSV files in ``outputs/``.

    python scripts/run_benchmark.py --benchmark gaia --method helm --data path/to/metadata.jsonl --mock --limit 5
    python scripts/run_benchmark.py --benchmark mind2web --method flat_rag --data data/mind2web --seeds 0 42 123 --llm-base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B

Two files are appended per benchmark: ``bench_<benchmark>.csv`` (one row per seed and reporting group, plus a ``mean``
row with standard errors across seeds) and ``bench_<benchmark>_episodes.csv`` (one row per episode). Every row carries
a ``source`` column: ``mock`` whenever a MockLLM / stand-in embedder was involved (forced by ``--mock``), ``measured``
only for a real model client. Run from inside ``code/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.benchmarks import BENCHMARKS, RETRIEVAL_ONLY, get_adapter_module  # noqa: E402
from helm.benchmarks.base import (  # noqa: E402
    BenchmarkError,
    MissingDataError,
    MissingDependencyError,
    ResultsWriter,
    embedder_source,
    episode_row,
    llm_source,
)
from helm.benchmarks.harness import make_agent  # noqa: E402
from helm.config import HelmConfig  # noqa: E402
from helm.methods import available  # noqa: E402
from helm.metrics import mean_se  # noqa: E402

AGENT_METHODS = [*available(), "none"]
RETRIEVAL_METHODS = ["bm25", "dense", "shnm"]
_IDENT = {"benchmark", "method", "seed", "split", "source", "model", "data", "embedding", "limit", "cfg_hash"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--method", required=True, help=f"one of helm.methods.available() or 'none' ({', '.join(AGENT_METHODS)}); "
                    "bright/qa also accept bm25, dense, shnm and any HELM variant")
    ap.add_argument("--data", help="benchmark data path (format per adapter, see its module docstring)")
    ap.add_argument("--corpus", help="qa only: passage corpus (jsonl of {id, text}) to retrieve from")
    ap.add_argument("--limit", type=int, help="max tasks (mind2web: per split)")
    ap.add_argument("--split", help="benchmark split (mind2web: CT/CW/CD or test_task/test_website/test_domain)")
    ap.add_argument("--seeds", type=int, nargs="+", help="default: cfg.seeds (0 42 123)")
    llm = ap.add_argument_group("reasoner (exactly one of --mock / --llm-base-url)")
    llm.add_argument("--mock", action="store_true", help="offline scripted MockLLM; rows are labelled source=mock")
    llm.add_argument("--llm-base-url", help="OpenAI-compatible endpoint, e.g. a local vLLM server")
    llm.add_argument("--model", help="model name served at the endpoint (default: cfg.reasoner.model)")
    llm.add_argument("--api-key-env", help="environment variable holding the API key (default: cfg.reasoner.api_key_env)")
    ap.add_argument("--config", help="HelmConfig YAML (defaults to the paper's reproducibility table)")
    ap.add_argument("--embedding", choices=["hashing", "sentence_transformers"], help="override cfg.embedding.backend")
    ap.add_argument("--embedding-model", help="override cfg.embedding.model_name")
    ap.add_argument("--max-steps", type=int, help="agent call budget per episode (scaffold knob, same for all methods)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3], help="gaia: levels to keep (paper: 2 3)")
    ap.add_argument("--file-root", help="gaia: directory with the task attachments (read_file tool)")
    ap.add_argument("--top-k", type=int, default=10, help="mind2web: candidates shown per step")
    ap.add_argument("--prefetch-recall", action="store_true", help="streambench: recall with x_t before the first model call")
    ap.add_argument("--no-keep-memory", action="store_true", help="streambench/lifelong: reset memory for every item")
    ap.add_argument("--env-factory", help="lifelong: 'package.module:callable' returning a LifelongEnv")
    ap.add_argument("--replay-k", type=int, help="lifelong: Exp=k replay baseline (k most recent successful trajectories)")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------------- construction
def resolve_config(args: argparse.Namespace) -> HelmConfig:
    if args.config and not Path(args.config).is_file():
        raise MissingDataError(f"config file not found: {args.config}")
    cfg = HelmConfig.from_yaml(args.config) if args.config else HelmConfig()
    over: Dict[str, Any] = {}
    if args.model:
        over["reasoner.model"] = args.model
    if args.llm_base_url:
        over["reasoner.base_url"] = args.llm_base_url
    if args.api_key_env:
        over["reasoner.api_key_env"] = args.api_key_env
    if args.embedding:
        over["embedding.backend"] = args.embedding
    if args.embedding_model:
        over["embedding.model_name"] = args.embedding_model
    return cfg.override(**over) if over else cfg


def check_args(args: argparse.Namespace) -> None:
    b = args.benchmark
    if b not in RETRIEVAL_ONLY and bool(args.mock) == bool(args.llm_base_url):  # bright uses no LLM
        raise BenchmarkError("choose exactly one reasoner: --mock (offline, labelled source=mock) or --llm-base-url URL [--model NAME]")
    if args.data is None:
        raise MissingDataError(f"--data is required for '{b}' (this repo ships no benchmark data; see helm/benchmarks/{BENCHMARKS[b].split('.')[-1]}.py for the format)")
    if args.data is not None and not Path(args.data).exists():
        raise MissingDataError(f"--data path does not exist: {args.data}")
    if b in ("bright", "qa"):
        try:
            from helm.benchmarks.bright import variant_overrides

            if args.method not in (*RETRIEVAL_METHODS, "none", "flat_rag"):
                variant_overrides(args.method)
        except KeyError:
            raise BenchmarkError(f"'{args.method}' is not valid for {b}: use bm25, dense, shnm, none (qa) or a HELM variant") from None
        if args.method == "none" and b == "bright":
            raise BenchmarkError("'none' is not a retrieval method for bright")
        if b == "qa" and args.method != "none" and not args.corpus:
            raise MissingDataError("qa with a retrieval method needs --corpus PATH (jsonl of {id, text}); use --method none for closed-book")
    elif args.method not in AGENT_METHODS:
        raise BenchmarkError(f"unknown method '{args.method}'. available: {AGENT_METHODS}")
    if b == "lifelong" and not args.env_factory:
        raise MissingDependencyError("lifelong needs --env-factory 'package.module:callable' returning a LifelongEnv adapted from the official DB/OS/KG environments")


def load_factory(spec: str):
    mod, _, attr = spec.partition(":")
    if not attr:
        raise BenchmarkError(f"--env-factory must look like package.module:callable, got '{spec}'")
    try:
        return getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as e:
        raise MissingDependencyError(f"cannot import --env-factory '{spec}': {e}") from e


def make_llm(args: argparse.Namespace, cfg: HelmConfig, mod: Any):
    if args.mock:
        return mod.mock_llm()
    from helm.llm.client import OpenAICompatClient

    try:
        return OpenAICompatClient.from_config(cfg.reasoner)
    except ImportError as e:
        raise MissingDependencyError("the 'openai' package is required for --llm-base-url (pip install openai)") from e


def make_embedder(cfg: HelmConfig):
    from helm.embedding.base import build_embedder

    try:
        return build_embedder(cfg.embedding)
    except ImportError as e:
        raise MissingDependencyError(f"embedding backend '{cfg.embedding.backend}' needs an optional package: {e}") from e


def make_adapter(args: argparse.Namespace, cfg: HelmConfig, embedder: Any, mod: Any, seed: int):
    """A fresh adapter per seed (envs and retrievers are stateful)."""
    b = args.benchmark
    if b == "gaia":
        return mod.GaiaAdapter(levels=tuple(args.levels), file_root=args.file_root, max_steps=args.max_steps)
    if b == "webarena":
        return mod.WebArenaAdapter(max_steps=args.max_steps or 30)
    if b == "mind2web":
        return mod.Mind2WebAdapter(top_k=args.top_k)
    if b == "streambench":
        return mod.StreamBenchAdapter(keep_memory=not args.no_keep_memory, prefetch=args.prefetch_recall, max_steps=args.max_steps or 6)
    if b == "lifelong":
        env = load_factory(args.env_factory)()
        replay = mod.ReplayMemory(args.replay_k) if args.replay_k else None
        return mod.LifelongAdapter(env, replay, keep_memory=not args.no_keep_memory, max_steps=args.max_steps or 15)
    if b == "bright":
        return mod.BrightAdapter()
    if b == "qa":
        if args.method == "none":
            return mod.QAAdapter()
        from helm.benchmarks.bright import build_retriever

        return mod.QAAdapter(build_retriever(args.method, embedder, cfg, seed=seed), mod.load_corpus(args.corpus))
    raise AssertionError(b)


def make_runner(args: argparse.Namespace, cfg: HelmConfig, embedder: Any, mod: Any, seed: int):
    """The object ``adapter.run_all`` drives: a retriever (bright) or an ``Agent``; plus the provenance label."""
    if args.benchmark == "bright":
        from helm.benchmarks.bright import build_retriever

        src = "mock" if args.mock else ("measured" if args.method == "bm25" else embedder_source(embedder))
        return build_retriever(args.method, embedder, cfg, seed=seed), src
    llm = make_llm(args, cfg, mod)
    kw: Dict[str, Any] = {"max_steps": args.max_steps} if args.max_steps else {}
    method = "none" if args.benchmark == "qa" else args.method  # qa's method is the retriever, not a memory backend
    agent = make_agent(method, cfg, llm=llm, seed=seed, **kw)
    return agent, ("mock" if args.mock else llm_source(llm))


# --------------------------------------------------------------------------------------------- reporting
def aggregate_rows(per_seed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean and standard error across seeds for every numeric column, per reporting group (``split``)."""
    out: List[Dict[str, Any]] = []
    for split in dict.fromkeys(r.get("split", "") for r in per_seed):
        rows = [r for r in per_seed if r.get("split", "") == split]
        agg: Dict[str, Any] = {k: rows[0][k] for k in _IDENT if k in rows[0]}
        agg.update(seed="mean", split=split, n_seeds=len(rows))
        for k in rows[0]:
            if k in _IDENT or k.endswith("_ci_lo") or k.endswith("_ci_hi") or k in ("n", "n_tasks", "n_steps"):
                continue
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            if len(vals) == len(rows):
                m, se = mean_se([float(v) for v in vals])
                agg[k], agg[k + "_se"] = m, se
        out.append(agg)
    return out


def fmt_row(r: Dict[str, Any]) -> str:
    keys = [k for k in ("score", "success_rate", "ctu_per_task", "epi_err_pct", "ndcg@10", "recall@10", "ele_acc", "op_f1", "step_sr", "sr") if k in r]
    return f"  seed={r.get('seed')!s:<5} split={r.get('split', '')!s:<5} " + " ".join(
        f"{k}={r[k]:.4g}" if isinstance(r[k], (int, float)) else f"{k}={r[k]}" for k in keys
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        cfg = resolve_config(args)
        check_args(args)
        mod = get_adapter_module(args.benchmark)
        seeds = list(args.seeds) if args.seeds else list(cfg.seeds)
        embedder = make_embedder(cfg) if args.benchmark in ("bright", "qa") else None
        cfg_hash = hashlib.sha1(json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:10]
        print(json.dumps({"benchmark": args.benchmark, "method": args.method, "seeds": seeds, "args": vars(args), "config": cfg.to_dict(), "cfg_hash": cfg_hash},
                         indent=2, default=str))
        out = Path(args.out_dir)
        summary_w = ResultsWriter(out / f"bench_{args.benchmark}.csv")
        episode_w = ResultsWriter(out / f"bench_{args.benchmark}_episodes.csv")
        per_seed: List[Dict[str, Any]] = []
        for seed in seeds:
            adapter = make_adapter(args, cfg, embedder, mod, seed)
            tasks = adapter.load(args.data, args.split, args.limit)
            runner, source = make_runner(args, cfg, embedder, mod, seed)
            results = adapter.run_all(tasks, runner)
            ident = {
                "benchmark": args.benchmark, "method": args.method, "seed": seed, "source": source,
                "model": "mock" if args.mock else cfg.reasoner.model, "data": args.data or "",
                "embedding": cfg.embedding.backend, "limit": args.limit if args.limit is not None else "", "cfg_hash": cfg_hash,
            }
            rows = [{**ident, **r} for r in adapter.summarize(tasks, results, seed=seed)]
            per_seed.extend(rows)
            summary_w.append_many(rows)
            episode_w.append_many(episode_row(r, **ident) for r in results)
            print(f"seed {seed} [{source}]:")
            for r in rows:
                print(fmt_row(r))
        if len(seeds) > 1:
            agg = aggregate_rows(per_seed)
            summary_w.append_many(agg)
            print("mean over seeds:")
            for r in agg:
                print(fmt_row(r))
        print(f"wrote {summary_w.path} and {episode_w.path}")
        return 0
    except (BenchmarkError, KeyError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
