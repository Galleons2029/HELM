"""Tiny byte-level tokenizer and tiny random-init Qwen3-config models for CPU smoke tests.

Nothing here is a result: the models are random-init, a few 10^5 parameters, and exist so the training code paths
(bidirectional encoder + InfoNCE, on-policy distillation, GSPO) can be exercised end-to-end offline in seconds.
Everything produced from them must be labelled ``"source": "smoke"``.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from .data_synthesis import make_synthetic_controller_dataset
from .opd import completion_logprobs, encode_completions, encode_prompts, set_seed

SMOKE_SYSTEM_PROMPT = "Memory controller. Emit ops."  # short stand-in for SYSTEM_PROMPT so smoke sequences stay short


_PRE = re.compile(r" ?[A-Za-z_]+| ?\d+| ?[^A-Za-z_\d\s]+|\s")


class ByteTokenizer:
    """Byte-level tokenizer: ids 0/1/2 are pad/bos/eos, bytes map to ``3 + byte``.

    With no merges it is a pure byte tokenizer. ``train_bpe`` adds a handful of byte-pair merges (GPT-2 style, learned on
    a tiny corpus) purely to shorten sequences so CPU smoke runs stay fast; unseen text falls back to bytes. Implements the
    small HF-like surface the training code uses (``encode``, ``decode``, ``pad_token_id``, ``eos_token_id``,
    ``save_pretrained``).
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    padding_side = "right"
    _OFFSET = 3

    def __init__(self, merges: Sequence[Tuple[int, int]] = ()):
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self._rank = {m: i for i, m in enumerate(self.merges)}
        self._bytes: Dict[int, bytes] = {i + self._OFFSET: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self._bytes[256 + self._OFFSET + i] = self._bytes[a] + self._bytes[b]
        self.vocab_size = 256 + self._OFFSET + len(self.merges)
        self._cache: Dict[str, List[int]] = {}

    def __len__(self) -> int:
        return self.vocab_size

    @classmethod
    def train_bpe(cls, texts: Sequence[str], n_merges: int = 200) -> "ByteTokenizer":
        """Greedy byte-pair merges over whitespace-delimited pieces; deterministic (ties break on the smaller pair)."""
        freq = Counter(piece for t in texts for piece in _PRE.findall(t))
        words = {w: [b + cls._OFFSET for b in w.encode("utf-8")] for w in freq}
        merges: List[Tuple[int, int]] = []
        for k in range(n_merges):
            pairs: Counter = Counter()
            for w, ids in words.items():
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += freq[w]
            if not pairs:
                break
            best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            new_id = 256 + cls._OFFSET + k
            merges.append(best)
            for w, ids in words.items():
                words[w] = cls._apply(ids, best, new_id)
        return cls(merges)

    @staticmethod
    def _apply(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _encode_piece(self, piece: str) -> List[int]:
        hit = self._cache.get(piece)
        if hit is None:
            ids = [b + self._OFFSET for b in piece.encode("utf-8")]
            while len(ids) > 1:
                ranked = [(self._rank.get(p, 1 << 30), p) for p in zip(ids, ids[1:])]
                rank, pair = min(ranked)
                if rank == 1 << 30:
                    break
                ids = self._apply(ids, pair, 256 + self._OFFSET + rank)
            hit = self._cache[piece] = ids
        return hit

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = [i for piece in _PRE.findall(text) for i in self._encode_piece(piece)]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        data = b"".join(self._bytes[i] for i in ids if i in self._bytes)
        return data.decode("utf-8", errors="replace")

    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tiny_tokenizer.json").write_text(json.dumps({"type": "byte_bpe", "merges": self.merges}))

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ByteTokenizer":
        f = Path(path) / "tiny_tokenizer.json"
        if not f.exists():
            raise FileNotFoundError(f"no tiny_tokenizer.json in {path}")
        return cls(json.loads(f.read_text())["merges"])


def load_tokenizer(path: str) -> Any:
    """``ByteTokenizer`` for smoke checkpoints, otherwise ``transformers.AutoTokenizer``."""
    if (Path(path) / "tiny_tokenizer.json").exists():
        return ByteTokenizer.from_pretrained(path)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def tiny_qwen3_config(
    vocab_size: int = 259,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    intermediate_size: int = 128,
    max_position_embeddings: int = 2048,
    **overrides: Any,
) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=ByteTokenizer.pad_token_id,
        bos_token_id=ByteTokenizer.bos_token_id,
        eos_token_id=ByteTokenizer.eos_token_id,
        tie_word_embeddings=True,
        **overrides,
    )


def tiny_causal_lm(seed: int = 0, **sizes: Any) -> Qwen3ForCausalLM:
    """Random-init tiny Qwen3 causal LM (controller student/teacher/policy)."""
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(tiny_qwen3_config(**sizes))


def tiny_encoder(seed: int = 0, **sizes: Any) -> Qwen3Model:
    """Random-init tiny Qwen3 backbone without LM head (embedding model)."""
    torch.manual_seed(seed)
    return Qwen3Model(tiny_qwen3_config(**sizes))


def sft_warmstart(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[Dict[str, Any]],
    steps: int,
    batch_size: int = 16,
    lr: float = 3e-3,
    seed: int = 0,
    max_prompt_len: Optional[int] = None,
) -> List[float]:
    """Smoke-only supervised warm start on ``{"prompt", "completion"}`` records (completion + EOS); returns step losses.

    Used to turn a random-init tiny model into a non-trivial teacher (or initial policy). The paper does not describe
    any SFT stage; this is test scaffolding.
    """
    rng = random.Random(seed)
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    pad = tok.pad_token_id
    model.train()
    losses = []
    for _ in range(steps):
        batch = rng.sample(list(records), min(batch_size, len(records)))
        p_ids = encode_prompts(tok, [r["prompt"] for r in batch], max_prompt_len)
        c_ids = encode_completions(tok, [r["completion"] for r in batch])
        logp, mask = completion_logprobs(model, p_ids, c_ids, pad, 1.0, grad=True)
        loss = -(logp * mask).sum() / mask.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    model.zero_grad(set_to_none=True)
    model.eval()
    return losses


def build_smoke_controller_setup(
    seed: int = 0,
    n_records: int = 192,
    n_merges: int = 160,
    teacher_steps: int = 150,
    student_steps: int = 25,
    teacher_sizes: Optional[Dict[str, int]] = None,
    student_sizes: Optional[Dict[str, int]] = None,
    with_teacher: bool = True,
) -> Dict[str, Any]:
    """Tokenizer, synthetic controller records, an SFT'd teacher and a lightly SFT'd (hence imperfect) student.

    The tokenizer is a byte-level BPE trained on the synthetic prompts/completions. The teacher is larger than the student
    (default hidden 96 / 3 layers vs 64 / 2 layers). Returns a dict with keys
    ``tok, records, teacher, student, teacher_losses, student_losses`` (``teacher`` is ``None`` if ``with_teacher=False``).
    """
    records = make_synthetic_controller_dataset(n_records, seed=seed, system_prompt=SMOKE_SYSTEM_PROMPT)
    tok = ByteTokenizer.train_bpe([r["prompt"] + r["completion"] for r in records], n_merges)
    v = tok.vocab_size
    t_sizes = {"hidden_size": 96, "num_layers": 3, "intermediate_size": 192, **(teacher_sizes or {})}
    teacher, t_losses = None, []
    if with_teacher:
        teacher = tiny_causal_lm(seed + 1, vocab_size=v, **t_sizes)
        t_losses = sft_warmstart(teacher, tok, records, teacher_steps, seed=seed)
        for p in teacher.parameters():
            p.requires_grad_(False)
    student = tiny_causal_lm(seed + 2, vocab_size=v, **(student_sizes or {}))
    s_losses = sft_warmstart(student, tok, records, student_steps, seed=seed + 7)
    return {"tok": tok, "records": records, "teacher": teacher, "student": student,
            "teacher_losses": t_losses, "student_losses": s_losses}
