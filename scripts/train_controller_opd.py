#!/usr/bin/env python3
"""Stage 1: on-policy distillation (reverse KL) of the memory controller.

    python scripts/train_controller_opd.py --smoke      # tiny random-init student/teacher on CPU -> outputs/opd_smoke/
    python scripts/train_controller_opd.py --student Qwen/Qwen3-4B --teacher <stronger-model> \\
        --prompts outputs/controller_prompts.jsonl --out outputs/opd    # NOT verified in the offline test suite (needs GPUs)

Prompt records are JSONL lines with a ``prompt`` field (``helm.training.data_synthesis.controller_prompt`` format).
Without ``--prompts`` the non-smoke run falls back to ``make_synthetic_controller_dataset`` prompts, which are NOT drawn
from real agent trajectories: replace them with simulator-derived prompts before relying on the result. The teacher must
share the student's tokenizer vocabulary. Smoke metrics are labelled ``"source": "smoke"``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.training import dtype_kwargs, print_config, resolve_device, thirds, write_metrics  # noqa: E402

NOT_VERIFIED = "NOT VERIFIED in the offline test suite (needs GPUs and hub checkpoints); lr/batch values are assumptions."


def run_smoke(args) -> dict:
    import torch

    from helm.training.opd import OPDConfig, OPDTrainer, sample_rollouts
    from helm.training.data_synthesis import state_from_dict
    from helm.training.rewards import toy_task_reward
    from helm.training.tiny import build_smoke_controller_setup

    torch.set_num_threads(min(4, torch.get_num_threads()))
    S = build_smoke_controller_setup(seed=args.seed)
    tok, recs, teacher, student = S["tok"], S["records"], S["teacher"], S["student"]
    cfg = OPDConfig(iters=args.iters or 60, prompts_per_iter=8, samples_per_prompt=2, max_new_tokens=60, lr=3e-3,
                    seed=args.seed)
    print_config("train_controller_opd --smoke", {"opd": cfg.__dict__, "n_prompts": len(recs),
                                                  "teacher": "tiny 96x3, SFT'd on the heuristic controller",
                                                  "student": "tiny 64x2, lightly SFT'd"},
                 "random-init tiny models on synthetic prompts; NOT a result")

    def toy_score(model) -> float:  # fraction-of-heuristic-ops agreement, sampled at temperature 1 on 48 prompts
        roll = sample_rollouts(model, tok, [r["prompt"] for r in recs[:48]], 60)
        return sum(toy_task_reward(state_from_dict(recs[i]["state"]), roll.texts[i]) for i in range(48)) / 48

    before = {"teacher_toy_task_reward": toy_score(teacher), "student_toy_task_reward": toy_score(student)}
    trainer = OPDTrainer(student, teacher, tok, cfg)
    hist = trainer.fit([r["prompt"] for r in recs],
                       on_step=lambda r: print(f"iter {r['iter']:3d} mean reverse KL {r['mean_reverse_kl']:.3f} "
                                               f"len {r['mean_completion_len']:.1f} eos {r['eos_frac']:.2f}"))
    after = {"student_toy_task_reward": toy_score(student)}
    out = Path(args.out)
    student.save_pretrained(out / "student")
    tok.save_pretrained(out / "student")
    kl = [h["mean_reverse_kl"] for h in hist]
    first, last = thirds(kl)
    return {"config": cfg.__dict__, "history": hist, "kl_first_third": first, "kl_last_third": last,
            "teacher_sft_final_loss": S["teacher_losses"][-1], "student_sft_final_loss": S["student_losses"][-1],
            "toy_reward_before": before, "toy_reward_after": after, "checkpoint": str(out / "student")}


def run_real(args) -> dict:  # pragma: no cover - needs GPUs and hub weights
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from helm.training.data_synthesis import make_synthetic_controller_dataset, read_jsonl, wrap_controller_prompt
    from helm.training.opd import OPDConfig, OPDTrainer

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.student)
    student = AutoModelForCausalLM.from_pretrained(args.student, **dtype_kwargs(dtype)).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **dtype_kwargs(dtype)).to(device)
    if args.prompts:
        prompts = [r["prompt"] for r in read_jsonl(args.prompts)]
        if not args.no_wrap_system_prompt:  # train on the exact prompt framing HFController uses at inference
            prompts = [wrap_controller_prompt(p) for p in prompts]
    else:
        print("WARNING: no --prompts given; using synthetic controller prompts (not real agent trajectories)")
        prompts = [r["prompt"] for r in make_synthetic_controller_dataset(args.n_synthetic_prompts, seed=args.seed)]
    cfg = OPDConfig(iters=args.iters or 100, prompts_per_iter=args.prompts_per_iter,
                    samples_per_prompt=args.samples_per_prompt, max_new_tokens=args.max_new_tokens,
                    max_prompt_len=args.max_prompt_len, lr=args.lr, micro_batch_size=args.micro_batch_size,
                    seed=args.seed)
    print_config("train_controller_opd", {"student": args.student, "teacher": args.teacher, "device": device,
                                          "dtype": args.dtype, "opd": cfg.__dict__, "n_prompts": len(prompts)},
                 NOT_VERIFIED)
    hist = OPDTrainer(student, teacher, tok, cfg).fit(
        prompts, on_step=lambda r: print(f"iter {r['iter']} mean reverse KL {r['mean_reverse_kl']:.4f}"))
    out = Path(args.out)
    student.save_pretrained(out / "student")
    tok.save_pretrained(out / "student")
    kl = [h["mean_reverse_kl"] for h in hist]
    first, last = thirds(kl)
    return {"config": cfg.__dict__, "history": hist, "kl_first_third": first, "kl_last_third": last,
            "student": args.student, "teacher": args.teacher, "checkpoint": str(out / "student")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--student", default="Qwen/Qwen3-4B")
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B", help="ASSUMPTION: any stronger model sharing the vocabulary")
    ap.add_argument("--prompts", default=None, help="JSONL with a `prompt` field")
    ap.add_argument("--no-wrap-system-prompt", action="store_true",
                    help="keep --prompts as given (default: wrap bare render_state prompts like HFController does)")
    ap.add_argument("--n-synthetic-prompts", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--prompts-per-iter", type=int, default=8)
    ap.add_argument("--samples-per-prompt", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-prompt-len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out = args.out or ("outputs/opd_smoke" if args.smoke else "outputs/opd")

    result = run_smoke(args) if args.smoke else run_real(args)
    path = write_metrics(Path(args.out) / "metrics.json", "smoke" if args.smoke else "measured", **result)
    print(f"\nmean reverse KL: first third {result['kl_first_third']:.3f} -> last third {result['kl_last_third']:.3f}")
    if args.smoke:
        print("toy task reward before:", result["toy_reward_before"], "after:", result["toy_reward_after"])
    print(f"metrics -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
