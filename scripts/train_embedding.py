#!/usr/bin/env python3
"""Tool-aware embedding fine-tuning: bidirectional decoder-only encoder + InfoNCE with epistemic counterfactuals.

    python scripts/train_embedding.py --smoke        # tiny random-init encoder on CPU, writes outputs/embedding_smoke/
    python scripts/train_embedding.py --data outputs/synth/triplets.jsonl --model Qwen/Qwen3-Embedding-4B \\
        --out outputs/embedding_ft                   # NOT verified in the offline test suite (needs GPUs and the hub checkpoint)

Defaults for the real run (tau 0.02, lr 1e-5, cosine schedule with 500 warmup steps, batch 256, 3 epochs, full fine-tuning)
are the values used for the paper's embedding model; they are not repeated in the method section. Full fine-tuning of the 4B model at batch 256 needs
multi-GPU memory sharding that this single-process script does not configure (use ``--micro-batch-size`` and
``--gradient-checkpointing`` to fit one large GPU). Smoke metrics are labelled ``"source": "smoke"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm.training import print_config, resolve_device, thirds, write_metrics  # noqa: E402

NOT_VERIFIED = "NOT VERIFIED in the offline test suite (needs GPUs and the hub checkpoint); defaults follow the paper's training setup."


def run_smoke(args) -> dict:
    import numpy as np
    import torch

    from helm.training.data_synthesis import SynthConfig, generate_offline_dataset, write_jsonl
    from helm.training.embedding_ft import EmbeddingFTConfig, EmbeddingTrainer, TextEncoder, evaluate_epistemic
    from helm.training.tiny import ByteTokenizer, tiny_encoder

    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = Path(args.out)
    records = generate_offline_dataset(
        SynthConfig(n=args.smoke_n, seed=args.seed, long_frac=0.2, min_ctx_tokens=160, max_ctx_tokens=300)
    )
    write_jsonl(out / "data.jsonl", records)
    held_out, train = records[:24], records[24:]  # evaluation records are never trained on
    texts = [t for r in train for t in [r["query"], r["positive"]] + [n["text"] for n in r["negatives"]]]
    tok = ByteTokenizer.train_bpe(texts, 300)
    enc = TextEncoder(tiny_encoder(args.seed, vocab_size=tok.vocab_size), tok, max_length=320)
    cfg = EmbeddingFTConfig(tau=0.05, lr=2e-3, warmup_steps=3, batch_size=16, epochs=5, micro_batch_size=None,
                            seed=args.seed)
    print_config("train_embedding --smoke", {"encoder": enc.meta(), "train": cfg.__dict__, "n_train": len(train),
                                             "n_held_out": len(held_out)},
                 "tiny random-init encoder, byte-level BPE tokenizer; NOT a result")
    before = evaluate_epistemic(enc, held_out)
    hist = EmbeddingTrainer(enc, cfg).fit(train, on_step=lambda r: print(f"step {r['step']:3d} loss {r['loss']:.3f} acc {r['acc']:.2f}"))
    after = evaluate_epistemic(enc, held_out)
    enc.save(out / "final")
    reloaded = TextEncoder.load(out / "final")
    probe = [held_out[0]["positive"], held_out[1]["positive"]]
    reload_diff = float(np.abs(enc.encode(probe) - reloaded.encode(probe)).max())
    losses = [h["loss"] for h in hist]
    first, last = thirds(losses)
    return {
        "encoder": enc.meta(), "config": cfg.__dict__, "n_train": len(train), "n_held_out": len(held_out),
        "history": hist, "loss_first_third": first, "loss_last_third": last,
        "held_out_before": before, "held_out_after": after, "reload_max_abs_diff": reload_diff,
        "checkpoint": str(out / "final"),
    }


def run_real(args) -> dict:  # pragma: no cover - needs GPUs and hub weights
    from helm.training.data_synthesis import read_jsonl
    from helm.training.embedding_ft import EmbeddingFTConfig, EmbeddingTrainer, evaluate_epistemic, load_pretrained_encoder

    records = read_jsonl(args.data)
    device = resolve_device(args.device)
    cfg = EmbeddingFTConfig(tau=args.tau, lr=args.lr, warmup_steps=args.warmup_steps, batch_size=args.batch_size,
                            epochs=args.epochs, micro_batch_size=args.micro_batch_size,
                            n_counterfactuals=args.n_counterfactuals, seed=args.seed)
    enc = load_pretrained_encoder(args.model, args.pooling, not args.causal, args.max_length, args.dtype, device,
                                  gradient_checkpointing=args.gradient_checkpointing)
    print_config("train_embedding", {"model": args.model, "device": device, "dtype": args.dtype, "encoder": enc.meta(),
                                     "train": cfg.__dict__, "n_records": len(records)}, NOT_VERIFIED)
    n_eval = min(args.eval_records, len(records) // 10)
    held_out, train = records[:n_eval], records[n_eval:]
    before = evaluate_epistemic(enc, held_out) if held_out else {}
    hist = EmbeddingTrainer(enc, cfg).fit(train, on_step=lambda r: print(f"step {r['step']} loss {r['loss']:.4f}"))
    after = evaluate_epistemic(enc, held_out) if held_out else {}
    enc.save(Path(args.out) / "final")
    return {"encoder": enc.meta(), "config": cfg.__dict__, "history": hist, "held_out_before": before,
            "held_out_after": after, "checkpoint": str(Path(args.out) / "final")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=168, help="smoke: records generated (24 are held out)")
    ap.add_argument("--data", help="JSONL from scripts/synthesize_data.py")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch-size", type=int, default=None, help="gradient-cached chunk size (sequences)")
    ap.add_argument("--n-counterfactuals", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--pooling", choices=("weighted", "mean"), default="weighted")
    ap.add_argument("--causal", action="store_true", help="keep the causal mask (ablation); default removes it")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--eval-records", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out = args.out or ("outputs/embedding_smoke" if args.smoke else "outputs/embedding_ft")
    if not args.smoke and not args.data:
        ap.error("--data is required unless --smoke")

    result = run_smoke(args) if args.smoke else run_real(args)
    path = write_metrics(Path(args.out) / "metrics.json", "smoke" if args.smoke else "measured", **result)
    if args.smoke:
        print(f"\nloss first third {result['loss_first_third']:.3f} -> last third {result['loss_last_third']:.3f}")
        print("held-out before:", json.dumps(result["held_out_before"]))
        print("held-out after: ", json.dumps(result["held_out_after"]))
        print(f"checkpoint reload max |diff| = {result['reload_max_abs_diff']:.2e}")
    print(f"metrics -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
