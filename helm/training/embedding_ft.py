"""Tool-aware embedding fine-tuning (Section "Tool-Aware Embedding Optimization", Appendix "Tool-aware embedding
fine-tuning details").

A decoder-only encoder (``Qwen3-Embedding-4B`` in the paper) is turned bidirectional by removing the causal mask, hidden
states are pooled by weighted mean, queries carry an instruction prefix, and the model is trained with InfoNCE over
in-batch negatives plus the synthesized epistemic counterfactuals (``helm.training.data_synthesis``).

Paper-stated: causal mask removed (bidirectional attention over tool invocations and results); weighted mean pooling of
final hidden states; instruction prepended; ``L = -log exp(s(m+)/tau) / (exp(s(m+)/tau) + sum_{m-} exp(s(m-)/tau))`` with
negatives = in-batch + epistemic counterfactuals; full fine-tuning of Qwen3-Embedding-4B.
Training hyper-parameters (not part of the method description): ``tau = 0.02``, ``lr = 1e-5``, cosine schedule with 500 warmup steps,
batch size 256, 3 epochs.

How the causal mask is removed (verified on transformers 5.9.0 / torch 2.12, ``tests/test_training_smoke.py``): in
transformers 5.x ``config.is_causal`` is read by a ``forward`` decorator, forwarded to the attention kernels and used by
``create_causal_mask`` to build a bidirectional mask instead. ``set_bidirectional`` flips that flag (and the per-layer
``is_causal`` attributes read by older versions). Because this differs across versions, ``TextEncoder`` never assumes it
works: it probes the model (does the hidden state at position 0 change when later tokens change, with and without
padding?) and, if the flag is ineffective, falls back to passing an explicit ``(B, 1, L, L)`` additive bidirectional mask
(``mask_mode="explicit"``); if neither is bidirectional it raises.

Verified only on tiny random-init Qwen3-config models on CPU. Fine-tuning the real ``Qwen3-Embedding-4B`` (full
fine-tuning, batch 256, 8k-token sequences) needs multi-GPU memory sharding that this single-process trainer does not set
up (it offers ``micro_batch_size`` gradient caching and gradient checkpointing, but that path is NOT verified on this
machine). Loading the hub checkpoint with ``AutoModel`` is likewise unverified (no network here).

ASSUMPTIONS (not specified in the paper):
  * Pooling weights are position-weighted (SGPT style, weight = 1-based token position among real tokens); plain mean is
    available via ``pooling="mean"``.
  * Queries are wrapped as ``Instruct: {instruction}\\nQuery:{text}`` (the Qwen3-Embedding convention, same as
    ``helm.embedding.st_backend``); documents/memories are embedded raw. No BOS/EOS special tokens are added.
  * Similarity is cosine; embeddings are L2-normalised before the temperature-scaled softmax.
  * In-batch candidates: every positive in the batch plus every counterfactual in the batch (so other queries'
    counterfactuals act as extra negatives). In-batch *positives* of records from the same seed (``seed_id``) are masked,
    since the three layers of one triplet are all relevant to one another (``mask_same_seed``).
  * AdamW with weight decay 0.01, gradient clipping 1.0, final cosine value 0 (min ratio 0), ``max_length`` 512 by default.
  * ``micro_batch_size`` enables two-pass gradient caching so the InfoNCE loss still sees the whole batch.
  * Pure-bf16 training when ``dtype="bfloat16"`` (no fp32 master weights).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SENTENCE_TRANSFORMERS_NOTE = """\
Using a checkpoint written by helm.training.embedding_ft
========================================================
(UNVERIFIED: sentence-transformers is not installed in the offline test environment.)

1. Native (tested): ``from helm.training.embedding_ft import TextEncoder; enc = TextEncoder.load(path)``; ``enc.encode(texts)``
   for memories, ``enc.encode(queries, instruction=...)`` for queries. This object satisfies helm.embedding.base.Embedder.

2. sentence-transformers: the checkpoint is a plain transformers model directory (config.json has ``is_causal: false``).
   Build the pipeline by hand with weighted-mean pooling and L2 normalisation, and prefix queries yourself:

       from sentence_transformers import SentenceTransformer, models
       word = models.Transformer(path, max_seq_length=8192)
       pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="weightedmean")
       model = SentenceTransformer(modules=[word, pool, models.Normalize()])
       q = model.encode([f"Instruct: {instruction}\\nQuery:{text}" for text in queries])

   Check that your transformers version honours ``config.is_causal=False`` (see ``TextEncoder.probe_bidirectional``);
   if it does not, use option 1.
"""

QUERY_TEMPLATE = "Instruct: {instruction}\nQuery:{text}"  # ASSUMPTION: Qwen3-Embedding convention


def format_query(instruction: str, text: str) -> str:
    return QUERY_TEMPLATE.format(instruction=instruction, text=text)


# ------------------------------------------------------------------------------------------------ pure functions
def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str = "weighted") -> torch.Tensor:
    """Pool ``(B, L, d)`` hidden states into ``(B, d)``.

    ``weighted``: weights are the 1-based positions of the real tokens (SGPT); ``mean``: uniform over real tokens.
    Padding tokens always get weight 0, whatever the padding side.
    """
    m = attention_mask.to(torch.float32)
    if mode == "mean":
        w = m
    elif mode == "weighted":  # ASSUMPTION: position-weighted (SGPT) form of the paper's "weighted mean pooling"
        w = torch.cumsum(m, dim=1) * m
    else:
        raise ValueError(f"unknown pooling mode: {mode}")
    return (hidden.float() * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-9)


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.0) -> float:
    """Learning-rate multiplier: linear warmup to 1 over ``warmup_steps`` then cosine decay to ``min_ratio``."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def info_nce_loss(
    q: torch.Tensor,
    pos: torch.Tensor,
    negs: Optional[torch.Tensor] = None,
    tau: float = 0.02,
    group_ids: Optional[Sequence[Any]] = None,
    cross_query_negatives: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """InfoNCE with in-batch and counterfactual negatives.

    ``q``/``pos``: ``(B, d)``; ``negs``: ``(B, K, d)`` counterfactuals of each query. All inputs are L2-normalised here.
    For query ``i`` the softmax runs over ``pos[i]`` (the positive), ``pos[j!=i]`` (in-batch negatives, skipped when
    ``group_ids[j] == group_ids[i]``), ``negs[i]`` (its own counterfactuals) and, if ``cross_query_negatives``,
    ``negs[j!=i]``. This is exactly ``-log exp(s+/tau) / (exp(s+/tau) + sum_neg exp(s-/tau))``.
    """
    q, pos = F.normalize(q.float(), dim=-1), F.normalize(pos.float(), dim=-1)
    B = q.size(0)
    logits = q @ pos.T / tau  # (B, B): the diagonal holds the positives
    if group_ids is not None:
        index = {g: i for i, g in enumerate(dict.fromkeys(group_ids))}
        g = torch.tensor([index[x] for x in group_ids], device=q.device)
        same = (g[:, None] == g[None, :]) & ~torch.eye(B, dtype=torch.bool, device=q.device)
        logits = logits.masked_fill(same, float("-inf"))
    if negs is not None and negs.numel() > 0:
        K = negs.size(1)
        n = F.normalize(negs.float(), dim=-1)
        nl = torch.einsum("bd,jkd->bjk", q, n) / tau  # (B, B, K)
        if not cross_query_negatives:
            own = torch.eye(B, dtype=torch.bool, device=q.device)[:, :, None].expand(-1, -1, K)
            nl = nl.masked_fill(~own, float("-inf"))
        logits = torch.cat([logits, nl.reshape(B, B * K)], dim=1)
    labels = torch.arange(B, device=q.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        stats = {
            "acc": float((logits.argmax(1) == labels).float().mean()),
            "pos_sim": float((q * pos).sum(-1).mean()),
        }
    return loss, stats


def bidirectional_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``(B, 1, L, L)`` additive mask: 0 where the key token is real, ``finfo.min`` where it is padding (no causal part)."""
    B, L = attention_mask.shape
    allowed = attention_mask.bool()[:, None, None, :].expand(B, 1, L, L)
    return torch.zeros((B, 1, L, L), dtype=dtype, device=attention_mask.device).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )


def set_bidirectional(backbone: torch.nn.Module, enabled: bool = True) -> None:
    """Flip the attention-causality flags a transformers model reads (config and per-layer attributes)."""
    backbone.config.is_causal = not enabled
    for m in backbone.modules():
        if hasattr(m, "is_causal"):
            m.is_causal = not enabled


# ------------------------------------------------------------------------------------------------ encoder
class TextEncoder(torch.nn.Module):
    """Bidirectional decoder-only encoder with weighted-mean pooling; satisfies ``helm.embedding.base.Embedder``."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        tokenizer: Any,
        pooling: str = "weighted",
        bidirectional: bool = True,
        max_length: int = 512,  # ASSUMPTION: the paper only states 8192 as the longest synthesized trajectory
        base_model: str = "",
    ):
        super().__init__()
        self.backbone, self.tokenizer = backbone, tokenizer
        self.pooling, self.max_length, self.base_model = pooling, max_length, base_model
        self.bidirectional = bidirectional
        self.mask_mode = "config"
        pool_hidden(torch.zeros(1, 1, 1), torch.ones(1, 1), pooling)  # validates the mode early
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        if bidirectional:
            self._negotiate_bidirectional()
        else:
            set_bidirectional(backbone, False)

    # --------------------------------------------------------------- properties
    @property
    def dim(self) -> int:
        return int(self.backbone.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    # --------------------------------------------------------------- causal-mask handling
    def _hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn = attention_mask
        if self.bidirectional and self.mask_mode == "explicit":
            attn = bidirectional_additive_mask(attention_mask, self.dtype)
        out = self.backbone(input_ids=input_ids, attention_mask=attn, use_cache=False)
        return out.last_hidden_state

    @torch.no_grad()
    def probe_bidirectional(self, seq_len: int = 8, tol: float = 1e-5) -> bool:
        """True iff the hidden state at position 0 depends on later tokens, both without and with padding."""
        was_training = self.training
        self.eval()
        g = torch.Generator().manual_seed(1234)
        V = int(self.backbone.config.vocab_size)
        a = torch.randint(0, V, (1, seq_len), generator=g)
        b = a.clone()
        b[0, seq_len // 2 :] = torch.randint(0, V, (seq_len - seq_len // 2,), generator=g)
        ok = True
        for pad in (0, 2):
            ids_a, ids_b = a.clone(), b.clone()
            mask = torch.ones(1, seq_len, dtype=torch.long)
            if pad:
                mask[0, -pad:] = 0
            ids_a, ids_b, mask = (t.to(self.device) for t in (ids_a, ids_b, mask))
            ha = self._hidden(ids_a, mask)[0, 0].float()
            hb = self._hidden(ids_b, mask)[0, 0].float()
            ok = ok and bool((ha - hb).abs().max() > tol)
        self.train(was_training)
        return ok

    def _negotiate_bidirectional(self) -> None:
        for mode in ("config", "explicit"):
            self.mask_mode = mode
            set_bidirectional(self.backbone, True)
            if self.probe_bidirectional():
                return
        raise RuntimeError(
            "could not make the backbone bidirectional: neither config.is_causal=False nor an explicit 4D mask changed "
            "the position-0 hidden state"
        )

    # --------------------------------------------------------------- tokenisation / encoding
    def tokenize(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-padded ``(input_ids, attention_mask)``; texts are truncated to ``max_length`` tokens (end is cut)."""
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        seqs = [list(self.tokenizer.encode(t, add_special_tokens=False))[: self.max_length] or [pad] for t in texts]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return ids.to(self.device), mask.to(self.device)

    def embed_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """L2-normalised float32 embeddings ``(B, dim)``."""
        pooled = pool_hidden(self._hidden(input_ids, attention_mask), attention_mask, self.pooling)
        return F.normalize(pooled, dim=-1)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:  # type: ignore[override]
        return self.embed_ids(*self.tokenize(texts))

    @torch.no_grad()
    def encode(self, texts: Sequence[str], instruction: Optional[str] = None, batch_size: int = 32) -> np.ndarray:
        """``helm.embedding.base.Embedder.encode``: queries (``instruction`` given) get the instruction prefix."""
        was_training = self.training
        self.eval()
        if instruction is not None:
            texts = [format_query(instruction, t) for t in texts]
        out = [self(list(texts[s : s + batch_size])).cpu().numpy() for s in range(0, len(texts), batch_size)]
        self.train(was_training)
        return np.concatenate(out, 0).astype(np.float32) if out else np.zeros((0, self.dim), dtype=np.float32)

    # --------------------------------------------------------------- persistence
    def meta(self) -> Dict[str, Any]:
        return {
            "pooling": self.pooling,
            "bidirectional": self.bidirectional,
            "mask_mode": self.mask_mode,
            "max_length": self.max_length,
            "dim": self.dim,
            "base_model": self.base_model,
            "query_template": QUERY_TEMPLATE,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        (p / "helm_embedding.json").write_text(json.dumps(self.meta(), indent=2))
        (p / "USAGE.md").write_text(SENTENCE_TRANSFORMERS_NOTE)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> "TextEncoder":
        from transformers import AutoModel

        from .tiny import load_tokenizer

        meta = json.loads((Path(path) / "helm_embedding.json").read_text())
        from . import dtype_kwargs

        backbone = AutoModel.from_pretrained(str(path), **(dtype_kwargs(dtype) if dtype else {}))
        if device:
            backbone.to(device)
        return cls(backbone, load_tokenizer(str(path)), meta["pooling"], meta["bidirectional"], meta["max_length"],
                   meta.get("base_model", ""))


def load_pretrained_encoder(
    model_name: str,
    pooling: str = "weighted",
    bidirectional: bool = True,
    max_length: int = 8192,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = False,
) -> TextEncoder:  # pragma: no cover - needs hub weights / GPUs
    """Hub checkpoint (e.g. ``Qwen/Qwen3-Embedding-4B``) -> ``TextEncoder``. NOT verified in the offline test suite."""
    from transformers import AutoModel, AutoTokenizer

    from . import dtype_kwargs

    kw: Dict[str, Any] = dtype_kwargs(getattr(torch, dtype))
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    backbone = AutoModel.from_pretrained(model_name, **kw)
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
    if device:
        backbone.to(device)
    return TextEncoder(backbone, AutoTokenizer.from_pretrained(model_name), pooling, bidirectional, max_length, model_name)


# ------------------------------------------------------------------------------------------------ training
@dataclass
class EmbeddingFTConfig:
    tau: float = 0.02
    lr: float = 1e-5
    warmup_steps: int = 500  # cosine schedule
    batch_size: int = 256
    epochs: int = 3
    micro_batch_size: Optional[int] = None  # ASSUMPTION: gradient-caching chunk (not in the paper)
    n_counterfactuals: int = 3  # task spec
    mask_same_seed: bool = True  # ASSUMPTION: same-triplet positives are not negatives of each other
    cross_query_negatives: bool = True  # ASSUMPTION: other queries' counterfactuals also act as negatives
    weight_decay: float = 0.01  # ASSUMPTION
    max_grad_norm: float = 1.0  # ASSUMPTION
    min_lr_ratio: float = 0.0  # ASSUMPTION: cosine decays to 0
    seed: int = 0


def _record_texts(rec: Dict[str, Any], k: int) -> Tuple[str, str, List[str]]:
    negs = [n["text"] for n in rec["negatives"]][:k]
    if len(negs) < k and negs:
        negs += [negs[i % len(negs)] for i in range(k - len(negs))]
    return format_query(rec["instruction"], rec["query"]), rec["positive"], negs


class EmbeddingTrainer:
    """Full fine-tuning loop with InfoNCE over in-batch + epistemic-counterfactual negatives."""

    def __init__(self, encoder: TextEncoder, cfg: EmbeddingFTConfig):
        self.enc, self.cfg = encoder, cfg
        params = [p for p in encoder.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.history: List[Dict[str, float]] = []
        self._step = 0

    # --------------------------------------------------------------- loss / gradient for one global batch
    def _encode_group(self, texts: Sequence[str], grad: bool, mb: Optional[int]) -> torch.Tensor:
        mb = mb or len(texts)
        outs = []
        with torch.set_grad_enabled(grad):
            for s in range(0, len(texts), mb):
                outs.append(self.enc(list(texts[s : s + mb])))
        return torch.cat(outs, 0)

    def loss_and_backward(self, recs: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        """Computes the InfoNCE loss on ``recs`` and accumulates parameter gradients (two-pass if ``micro_batch_size``)."""
        cfg = self.cfg
        rows = [_record_texts(r, cfg.n_counterfactuals) for r in recs]
        q_txt = [r[0] for r in rows]
        p_txt = [r[1] for r in rows]
        n_txt = [t for r in rows for t in r[2]]
        K = len(rows[0][2])
        groups = [r["seed_id"] for r in recs] if cfg.mask_same_seed else None
        B = len(recs)

        def loss_fn(Q: torch.Tensor, P: torch.Tensor, N: torch.Tensor):
            return info_nce_loss(Q, P, N.view(B, K, -1) if K else None, cfg.tau, groups, cfg.cross_query_negatives)

        if cfg.micro_batch_size is None or cfg.micro_batch_size >= max(len(q_txt), len(n_txt)):
            Q, P = self._encode_group(q_txt, True, None), self._encode_group(p_txt, True, None)
            N = self._encode_group(n_txt, True, None) if K else torch.zeros(0, Q.size(1), device=Q.device)
            loss, stats = loss_fn(Q, P, N)
            loss.backward()
        else:  # gradient caching: embeddings first (no grad), loss on them, then re-encode chunk-wise with grad
            mb = cfg.micro_batch_size
            Q, P = (self._encode_group(t, False, mb).requires_grad_() for t in (q_txt, p_txt))
            N = (self._encode_group(n_txt, False, mb) if K else torch.zeros(0, Q.size(1), device=Q.device)).requires_grad_()
            loss, stats = loss_fn(Q, P, N)
            loss.backward()
            for txt, leaf in ((q_txt, Q), (p_txt, P), (n_txt, N)):
                for s in range(0, len(txt), mb):
                    emb = self.enc(txt[s : s + mb])
                    emb.backward(leaf.grad[s : s + mb])
        return {"loss": float(loss.detach()), **stats}

    def train_step(self, recs: Sequence[Dict[str, Any]], lr_scale: float = 1.0) -> Dict[str, float]:
        self.enc.train()
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr * lr_scale
        self.opt.zero_grad(set_to_none=True)
        stats = self.loss_and_backward(recs)
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.enc.parameters(), self.cfg.max_grad_norm))
        self.opt.step()
        self._step += 1
        rec = {"step": self._step, "lr": self.cfg.lr * lr_scale, "grad_norm": gnorm, **stats}
        self.history.append(rec)
        return rec

    def steps_per_epoch(self, n: int) -> int:
        return max(1, n // self.cfg.batch_size)

    def fit(
        self, records: Sequence[Dict[str, Any]], on_step: Optional[Callable[[Dict[str, float]], None]] = None
    ) -> List[Dict[str, float]]:
        cfg = self.cfg
        rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        spe = self.steps_per_epoch(len(records))
        total = spe * cfg.epochs
        bs = min(cfg.batch_size, len(records))
        order = list(range(len(records)))
        for _ in range(cfg.epochs):
            rng.shuffle(order)
            for b in range(spe):
                batch = [records[i] for i in order[b * bs : (b + 1) * bs]]
                scale = cosine_with_warmup(self._step, cfg.warmup_steps, total, cfg.min_lr_ratio)
                rec = self.train_step(batch, scale)
                if on_step:
                    on_step(rec)
        return self.history

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate_epistemic(encoder: TextEncoder, records: Sequence[Dict[str, Any]], batch_size: int = 32) -> Dict[str, float]:
    """Diagnostics measured on ``records``.

    ``epistemic_acc``: fraction of (query, counterfactual) pairs with ``sim(q, positive) > sim(q, counterfactual)``.
    ``retrieval_top1``: fraction of queries whose own positive is the nearest among all positives in ``records``.
    """
    enc = encoder
    q = np.zeros((len(records), enc.dim), dtype=np.float32)
    for ins in {r["instruction"] for r in records}:  # queries are embedded per instruction
        idx = [i for i, r in enumerate(records) if r["instruction"] == ins]
        q[idx] = enc.encode([records[i]["query"] for i in idx], instruction=ins, batch_size=batch_size)
    pos = enc.encode([r["positive"] for r in records], batch_size=batch_size)
    neg_txt = [(i, n["text"]) for i, r in enumerate(records) for n in r["negatives"]]
    neg = enc.encode([t for _, t in neg_txt], batch_size=batch_size)
    s_pos = (q * pos).sum(-1)
    wins = [float(s_pos[i] > float(q[i] @ neg[j])) for j, (i, _) in enumerate(neg_txt)]
    sims = q @ pos.T
    return {
        "epistemic_acc": float(np.mean(wins)) if wins else float("nan"),
        "retrieval_top1": float((sims.argmax(1) == np.arange(len(records))).mean()),
        "mean_pos_sim": float(s_pos.mean()),
        "n_eval": float(len(records)),
    }
