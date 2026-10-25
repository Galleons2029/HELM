#!/usr/bin/env python3
"""Stage 2: Group Sequence Policy Optimization of the memory controller with the hybrid reward alpha*r_task + beta*r_mem.

    python scripts/train_controller_gspo.py --smoke     # tiny SFT'd policy + toy verifiable reward -> outputs/gspo_smoke/
    python scripts/train_controller_gspo.py --policy outputs/opd/student --prompts outputs/controller_records.jsonl \\
        --out outputs/gspo                              # NOT verified in the offline test suite (needs GPUs)

Records are JSONL lines with ``prompt`` and (for the built-in toy reward) ``state``. The task reward ``r_task`` must be the
benchmark metric; that comes from the simulator/harness, which this script does not contain. Plug it in with
``--reward-fn package.module:function`` where ``function(record, completion_text, n_completion_tokens)`` returns a float or
a dict with a ``reward`` key (plus optional ``r_task`` / ``r_mem`` for logging). The default ``toy`` reward only checks the
completion against the offline heuristic controller and is NOT a task metric. Clip ranges (3e-4 / 4e-4) and group size 8 are
assumptions (see helm/training/gspo.py). Smoke metrics are labelled ``"source": "smoke"``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.training import dtype_kwargs, print_config, resolve_device, thirds, write_metrics  # noqa: E402

NOT_VERIFIED = "NOT VERIFIED in the offline test suite (needs GPUs and hub checkpoints); lr/group/clip values are assumptions."


def load_reward_fn(spec: str, alpha: float, beta: float):
    from helm.training.rewards import make_toy_reward_fn

    if spec == "toy":
        return make_toy_reward_fn(alpha, beta)
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def run_smoke(args) -> dict:
    import torch

    from helm.training.gspo import GSPOConfig, GSPOTrainer
    from helm.training.rewards import make_toy_reward_fn
    from helm.training.tiny import build_smoke_controller_setup

    torch.set_num_threads(min(4, torch.get_num_threads()))
    S = build_smoke_controller_setup(seed=args.seed, student_steps=100, with_teacher=False)
    tok, recs, policy = S["tok"], S["records"], S["student"]
    pool = recs[:4]  # a small prompt pool is reused every iteration (keeps the toy run's reward estimate low-noise)
    cfg = GSPOConfig(iters=args.iters or 40, prompts_per_iter=4, group_size=16, max_new_tokens=48, lr=1e-3, seed=args.seed)
    reward_fn = make_toy_reward_fn(args.alpha, args.beta)
    print_config("train_controller_gspo --smoke", {"gspo": cfg.__dict__, "alpha": args.alpha, "beta": args.beta,
                                                   "n_prompt_pool": len(pool), "reward": "toy (heuristic-agreement)"},
                 "random-init tiny policy, SFT warm start, toy reward; NOT a result")
    hist = GSPOTrainer(policy, tok, reward_fn, cfg).fit(
        pool, on_step=lambda r: print(f"iter {r['iter']:3d} mean R {r['mean_reward']:.3f} r_task {r['mean_r_task']:.3f} "
                                      f"r_mem {r['mean_r_mem']:.3f} clip {r['clip_frac']:.2f}"))
    out = Path(args.out)
    policy.save_pretrained(out / "policy")
    tok.save_pretrained(out / "policy")
    first, last = thirds([h["mean_reward"] for h in hist])
    return {"config": cfg.__dict__, "alpha": args.alpha, "beta": args.beta, "history": hist,
            "reward_first_third": first, "reward_last_third": last, "sft_final_loss": S["student_losses"][-1],
            "checkpoint": str(out / "policy")}


def run_real(args) -> dict:  # pragma: no cover - needs GPUs and hub weights
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from helm.training.data_synthesis import make_synthetic_controller_dataset, read_jsonl, wrap_controller_prompt
    from helm.training.gspo import GSPOConfig, GSPOTrainer

    device = resolve_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.policy)
    policy = AutoModelForCausalLM.from_pretrained(args.policy, **dtype_kwargs(getattr(torch, args.dtype))).to(device)
    if args.prompts:
        records = read_jsonl(args.prompts)
        if not args.no_wrap_system_prompt:  # train on the exact prompt framing HFController uses at inference
            records = [{**r, "prompt": wrap_controller_prompt(r["prompt"])} for r in records]
    else:
        print("WARNING: no --prompts given; using synthetic controller records (not real agent trajectories)")
        records = make_synthetic_controller_dataset(args.n_synthetic_prompts, seed=args.seed)
    if args.reward_fn == "toy":
        print("WARNING: using the toy heuristic-agreement reward, NOT a benchmark task metric")
    cfg = GSPOConfig(iters=args.iters or 100, prompts_per_iter=args.prompts_per_iter, group_size=args.group_size,
                     clip_low=args.clip_low, clip_high=args.clip_high, n_minibatches=args.n_minibatches,
                     max_new_tokens=args.max_new_tokens, max_prompt_len=args.max_prompt_len, lr=args.lr,
                     micro_batch_size=args.micro_batch_size, seed=args.seed)
    print_config("train_controller_gspo", {"policy": args.policy, "device": device, "dtype": args.dtype,
                                           "gspo": cfg.__dict__, "alpha": args.alpha, "beta": args.beta,
                                           "reward_fn": args.reward_fn, "n_records": len(records)}, NOT_VERIFIED)
    hist = GSPOTrainer(policy, tok, load_reward_fn(args.reward_fn, args.alpha, args.beta), cfg).fit(
        records, on_step=lambda r: print(f"iter {r['iter']} mean R {r['mean_reward']:.4f}"))
    out = Path(args.out)
    policy.save_pretrained(out / "policy")
    tok.save_pretrained(out / "policy")
    first, last = thirds([h["mean_reward"] for h in hist])
    return {"config": cfg.__dict__, "alpha": args.alpha, "beta": args.beta, "history": hist,
            "reward_first_third": first, "reward_last_third": last, "policy_init": args.policy,
            "checkpoint": str(out / "policy")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--policy", default="Qwen/Qwen3-4B", help="HF name or Stage-1 checkpoint directory")
    ap.add_argument("--prompts", default=None, help="JSONL records with `prompt` (and `state` for the toy reward)")
    ap.add_argument("--no-wrap-system-prompt", action="store_true",
                    help="keep --prompts as given (default: wrap bare render_state prompts like HFController does)")
    ap.add_argument("--n-synthetic-prompts", type=int, default=512)
    ap.add_argument("--reward-fn", default="toy", help="`toy` or `package.module:function`")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight on r_task (paper: 1.0)")
    ap.add_argument("--beta", type=float, default=0.05, help="weight on r_mem (paper: 0.05)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--prompts-per-iter", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8, help="ASSUMPTION")
    ap.add_argument("--clip-low", type=float, default=3e-4, help="ASSUMPTION: verify against the GSPO paper")
    ap.add_argument("--clip-high", type=float, default=4e-4, help="ASSUMPTION: verify against the GSPO paper")
    ap.add_argument("--n-minibatches", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-prompt-len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-6, help="ASSUMPTION")
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out = args.out or ("outputs/gspo_smoke" if args.smoke else "outputs/gspo")

    result = run_smoke(args) if args.smoke else run_real(args)
    path = write_metrics(Path(args.out) / "metrics.json", "smoke" if args.smoke else "measured", **result)
    print(f"\nmean reward: first third {result['reward_first_third']:.3f} -> last third {result['reward_last_third']:.3f}")
    print(f"metrics -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
