"""Stage 1: on-policy distillation of the memory controller (Appendix "Stage 1 on-policy distillation details").

Recipe (Thinking-Machines style, as described in the paper): each iteration (1) samples completions from the student
``pi_theta``, (2) scores the *same sampled tokens* with a fixed teacher ``pi_T`` (no gradient through the teacher),
(3) sets the per-token advantage ``A_{t+1} = -(log pi_theta(a|s) - log pi_T(a|s))`` (detached, zero discount) and
(4) applies an importance-sampling policy-gradient update with those advantages. With one update per batch the
importance ratio is 1 and the expected gradient equals minus the gradient of the per-token reverse KL
``KL(pi_theta || pi_T)`` (checked numerically in ``tests/test_training_math.py``).

The module has two layers: pure tensor functions (``reverse_kl_advantages``, ``mean_reverse_kl``, ``opd_loss``) and the
HF-model plumbing (``sample_rollouts``, ``completion_logprobs``, ``teacher_logprobs``, ``OPDTrainer``). The plumbing is
shared with ``helm.training.gspo``.

Verified here only on tiny random-init Qwen3-config models on CPU (see ``tests/test_training_smoke.py``). Training the
real student (``Qwen/Qwen3-4B``) with a real teacher is NOT verified in the offline test suite.

ASSUMPTIONS (not in the paper): sampling temperature 1.0 with no top-k/top-p truncation (required for the on-policy
estimator; note that transformers' default ``top_k=50`` would silently break it, so it is overridden); learning rate
1e-5, AdamW, gradient clipping 1.0; 8 prompts x 2 samples per iteration; the teacher scores the same prompt as the student
unless ``teacher_prompts`` is given (then both must share the tokenizer vocabulary); no importance-ratio clipping.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------------ pure functions
def reverse_kl_advantages(student_logp: torch.Tensor, teacher_logp: torch.Tensor) -> torch.Tensor:
    """``A = -(log pi_theta(a|s) - log pi_T(a|s))`` per token, detached (neither policy receives gradient)."""
    return -(student_logp.detach() - teacher_logp.detach())


def mean_reverse_kl(student_logp: torch.Tensor, teacher_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sampled estimate of the per-token reverse KL: mean over valid tokens of ``log pi_theta - log pi_T``."""
    m = mask.to(student_logp.dtype)
    return ((student_logp.detach() - teacher_logp.detach()) * m).sum() / m.sum().clamp_min(1.0)


def opd_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    normalizer: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Importance-sampling policy-gradient loss ``-(exp(logp_new - logp_old) * A)`` averaged over valid tokens.

    ``normalizer`` defaults to the number of valid tokens; pass the whole-batch token count when accumulating
    gradients over micro-batches.
    """
    m = mask.to(logp_new.dtype)
    ratio = torch.exp(logp_new - logp_old.detach())
    total = -(ratio * advantages.detach() * m).sum()
    denom = m.sum() if normalizer is None else torch.as_tensor(normalizer, dtype=total.dtype, device=total.device)
    return total / denom.clamp_min(1.0)


# ------------------------------------------------------------------------------------------------ tokenisation helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def encode_prompts(tok: Any, prompts: Sequence[str], max_prompt_len: Optional[int] = None) -> List[List[int]]:
    """Token ids per prompt (no special tokens); over-long prompts keep their *end*."""
    out = []
    for p in prompts:
        ids = list(tok.encode(p, add_special_tokens=False))
        out.append(ids[-max_prompt_len:] if max_prompt_len else ids)
    return out


def encode_completions(tok: Any, completions: Sequence[str], add_eos: bool = True) -> List[List[int]]:
    out = []
    for c in completions:
        ids = list(tok.encode(c, add_special_tokens=False))
        out.append(ids + [tok.eos_token_id] if add_eos else ids)
    return out


@dataclass
class Rollout:
    """A batch of sampled completions (ids include the terminating EOS when the model emitted one)."""

    prompt_ids: List[List[int]]
    completion_ids: List[List[int]]
    texts: List[str]

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def lengths(self) -> List[int]:
        return [len(c) for c in self.completion_ids]


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _pad_cols(x: torch.Tensor, width: int) -> torch.Tensor:
    return F.pad(x, (0, width - x.size(1))) if x.size(1) < width else x


# ------------------------------------------------------------------------------------------------ sampling / scoring
@torch.no_grad()
def sample_rollouts(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    max_prompt_len: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Rollout:
    """Sample one completion per prompt from ``model``.

    Pure ancestral sampling (``top_k=0, top_p=1``): transformers' generation default ``top_k=50`` would otherwise truncate
    the distribution and the sampled tokens would no longer be on-policy.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for on-policy sampling")
    was_training = model.training
    model.eval()
    dev = _device_of(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt_ids = encode_prompts(tok, prompts, max_prompt_len)
    comp_ids: List[List[int]] = []
    bs = batch_size or len(prompt_ids)
    for s in range(0, len(prompt_ids), bs):
        chunk = prompt_ids[s : s + bs]
        width = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, p in enumerate(chunk):  # left padding for generation
            ids[i, width - len(p) :] = torch.tensor(p)
            att[i, width - len(p) :] = 1
        out = model.generate(
            input_ids=ids.to(dev),
            attention_mask=att.to(dev),
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad,
            eos_token_id=tok.eos_token_id,
        )
        for row in out[:, width:].tolist():
            if tok.eos_token_id in row:
                row = row[: row.index(tok.eos_token_id) + 1]
            comp_ids.append(row)
    model.train(was_training)
    texts = [tok.decode(c, skip_special_tokens=True) for c in comp_ids]
    return Rollout(prompt_ids, comp_ids, texts)


def _chunk_logprobs(
    model: torch.nn.Module,
    prompt_ids: Sequence[Sequence[int]],
    completion_ids: Sequence[Sequence[int]],
    pad_id: int,
    temperature: float,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dev = _device_of(model)
    B = len(prompt_ids)
    P = torch.tensor([len(p) for p in prompt_ids])
    Cn = torch.tensor([len(c) for c in completion_ids])
    L = int((P + Cn).max())
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        s = torch.tensor(list(p) + list(c))
        ids[i, : len(s)] = s
        att[i, : len(s)] = 1
    ar = torch.arange(width)[None, :]
    mask = ar < Cn[:, None]
    pos = (ar + P[:, None] - 1).clamp(0, L - 1)  # the hidden state at pos-1 predicts completion token j
    targets = ids.gather(1, (pos + 1).clamp(max=L - 1))
    ids, att, pos, targets, mask = (t.to(dev) for t in (ids, att, pos, targets, mask))
    decoder, head = getattr(model, "model", None), model.get_output_embeddings()
    if decoder is not None and head is not None:  # only project the completion positions through the LM head
        hid = decoder(input_ids=ids, attention_mask=att).last_hidden_state
        logits = head(hid.gather(1, pos[..., None].expand(-1, -1, hid.size(-1))))
    else:
        full = model(input_ids=ids, attention_mask=att).logits
        logits = full.gather(1, pos[..., None].expand(-1, -1, full.size(-1)))
    logp = F.log_softmax(logits.float() / temperature, dim=-1).gather(-1, targets[..., None]).squeeze(-1)
    return logp * mask, mask


def completion_logprobs(
    model: torch.nn.Module,
    prompt_ids: Sequence[Sequence[int]],
    completion_ids: Sequence[Sequence[int]],
    pad_id: int,
    temperature: float = 1.0,
    grad: bool = False,
    micro_batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(logp, mask)`` of shape ``(B, C)``: log-probability of each completion token given prompt and prefix.

    ``C`` is the longest completion; padded positions have ``mask == False`` and ``logp == 0``. With ``grad=False`` the
    batch may be split into micro-batches; with ``grad=True`` pass an already micro-sized batch.
    """
    width = max(len(c) for c in completion_ids)
    mb = micro_batch_size or len(prompt_ids)
    outs: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    with torch.set_grad_enabled(grad):
        for s in range(0, len(prompt_ids), mb):
            lp, m = _chunk_logprobs(model, prompt_ids[s : s + mb], completion_ids[s : s + mb], pad_id, temperature, width)
            outs.append(lp)
            masks.append(m)
    return torch.cat(outs, 0), torch.cat(masks, 0)


@torch.no_grad()
def teacher_logprobs(
    teacher: torch.nn.Module,
    rollout: Rollout,
    pad_id: int,
    temperature: float = 1.0,
    teacher_prompt_ids: Optional[Sequence[Sequence[int]]] = None,
    micro_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """``log pi_T`` of the student's sampled tokens (no gradient). ``teacher_prompt_ids`` overrides the prompt."""
    was_training = teacher.training
    teacher.eval()
    pids = teacher_prompt_ids if teacher_prompt_ids is not None else rollout.prompt_ids
    lp, _ = completion_logprobs(teacher, pids, rollout.completion_ids, pad_id, temperature, False, micro_batch_size)
    teacher.train(was_training)
    return lp


# ------------------------------------------------------------------------------------------------ trainer
@dataclass
class OPDConfig:
    # ASSUMPTION: every value in this config (the paper specifies the objective, not the optimisation hyper-parameters)
    iters: int = 100
    prompts_per_iter: int = 8
    samples_per_prompt: int = 2  # prompt reuse: several samples per prompt
    max_new_tokens: int = 64
    max_prompt_len: Optional[int] = None
    temperature: float = 1.0
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    updates_per_batch: int = 1
    micro_batch_size: Optional[int] = None
    seed: int = 0


class OPDTrainer:
    """Student/teacher on-policy distillation loop."""

    def __init__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        tok: Any,
        cfg: OPDConfig,
        teacher_prompt_fn: Optional[Callable[[str], str]] = None,
    ):
        self.student, self.teacher, self.tok, self.cfg = student, teacher, tok, cfg
        self.teacher_prompt_fn = teacher_prompt_fn
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        params = [p for p in student.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.history: List[Dict[str, float]] = []
        self._it = 0

    def step(self, prompts: Sequence[str]) -> Dict[str, float]:
        cfg = self.cfg
        rep = [p for p in prompts for _ in range(cfg.samples_per_prompt)]
        roll = sample_rollouts(self.student, self.tok, rep, cfg.max_new_tokens, cfg.temperature, cfg.max_prompt_len,
                               cfg.micro_batch_size)
        logp_old, mask = completion_logprobs(self.student, roll.prompt_ids, roll.completion_ids, self.pad,
                                             cfg.temperature, False, cfg.micro_batch_size)
        t_pids = None
        if self.teacher_prompt_fn is not None:
            t_pids = encode_prompts(self.tok, [self.teacher_prompt_fn(p) for p in rep], cfg.max_prompt_len)
        logp_t = teacher_logprobs(self.teacher, roll, self.pad, cfg.temperature, t_pids, cfg.micro_batch_size)
        logp_t = logp_t.to(logp_old.device)
        adv = reverse_kl_advantages(logp_old, logp_t)
        kl = float(mean_reverse_kl(logp_old, logp_t, mask))
        n_tok = mask.sum().clamp_min(1)

        self.student.train()
        loss_val, gnorm = 0.0, 0.0
        mb = cfg.micro_batch_size or len(roll)
        for _ in range(cfg.updates_per_batch):
            self.opt.zero_grad(set_to_none=True)
            loss_val = 0.0
            for s in range(0, len(roll), mb):
                sl = slice(s, s + mb)
                lp_new, m = completion_logprobs(self.student, roll.prompt_ids[sl], roll.completion_ids[sl], self.pad,
                                                cfg.temperature, True)
                width = lp_new.size(1)
                loss = opd_loss(lp_new, logp_old[sl, :width], adv[sl, :width], m, normalizer=n_tok)
                loss.backward()
                loss_val += float(loss.detach())
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.max_grad_norm))
            self.opt.step()
        self._it += 1
        eos = sum(1 for c in roll.completion_ids if c and c[-1] == self.tok.eos_token_id) / len(roll)
        rec = {
            "iter": self._it,
            "mean_reverse_kl": kl,
            "mean_advantage": -kl,
            "loss": loss_val,
            "grad_norm": gnorm,
            "n_tokens": float(n_tok),
            "mean_completion_len": float(sum(roll.lengths()) / len(roll)),
            "eos_frac": eos,
        }
        self.history.append(rec)
        return rec

    def fit(
        self,
        prompts: Sequence[str],
        iters: Optional[int] = None,
        on_step: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        rng = random.Random(self.cfg.seed)
        set_seed(self.cfg.seed)
        for _ in range(iters if iters is not None else self.cfg.iters):
            batch = rng.sample(list(prompts), min(self.cfg.prompts_per_iter, len(prompts)))
            rec = self.step(batch)
            if on_step:
                on_step(rec)
        return self.history

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)
