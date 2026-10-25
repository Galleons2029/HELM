"""Stage 2: Group Sequence Policy Optimization (GSPO) for the memory controller.

Per prompt, ``G`` completions are sampled and scored with the hybrid reward ``R = alpha * r_task + beta * r_mem``
(``helm.training.rewards``). Advantages are group-normalised, ``A_i = (R_i - mean) / (std + eps)``, the importance ratio
is *sequence level* and length normalised, ``s_i = exp( (1/|y_i|) * sum_t (log pi_theta(y_it) - log pi_old(y_it)) )``, and the
objective is the clipped surrogate ``-min(s_i A_i, clip(s_i, 1 - eps_low, 1 + eps_high) A_i)`` averaged over the group.
No KL penalty is used.

Verified here only on a tiny random-init Qwen3-config policy with a toy reward (``tests/test_training_smoke.py``). Training
the real controller (``Qwen/Qwen3-4B``) with a benchmark task reward is NOT verified in the offline test suite.

ASSUMPTIONS (the paper specifies only GSPO, the reward form and alpha/beta):
  * ``clip_low = 3e-4`` and ``clip_high = 4e-4``. The GSPO paper (Zheng et al., 2025) uses very small sequence-level clip
    ranges because sequence ratios concentrate near 1, but these two values are
    unconfirmed: check them against the original paper before relying on them.
  * ``group_size = 8``, 8 prompts per iteration, learning rate 1e-6, AdamW, gradient clipping 1.0, temperature 1.0, no
    top-k/top-p truncation, ``eps = 1e-6``.
  * Group standard deviation is the population std (``correction=0``).
  * A group whose rewards are all equal has zero advantage and is skipped in the backward pass (its gradient is exactly
    zero), but still counts in the loss normaliser, so skipping does not change the update.
  * ``n_minibatches`` > 1 splits the prompts of one rollout batch into sequential optimiser steps (so the sequence ratio
    departs from 1 and clipping can bind); the default of 1 is fully on-policy.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch

from .opd import Rollout, completion_logprobs, sample_rollouts, set_seed

RewardOut = Union[float, Dict[str, float]]
RewardFn = Callable[[Dict[str, Any], str, int], RewardOut]


# ------------------------------------------------------------------------------------------------ pure functions
def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``(R - mean) / (std + eps)`` along the last (group) dimension of an ``(N, G)`` reward tensor."""
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, correction=0)  # ASSUMPTION: population std
    return (rewards - mean) / (std + eps)


def sequence_ratio(logp_new: torch.Tensor, logp_old: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``s_i = exp(mean_t(logp_new - logp_old))`` over the valid tokens of each sequence; shape ``(B,)``."""
    m = mask.to(logp_new.dtype)
    log_ratio = ((logp_new - logp_old.detach()) * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    return torch.exp(log_ratio)


def gspo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float,
    clip_high: float,
    normalizer: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Clipped sequence-level surrogate, summed over sequences and divided by ``normalizer`` (default: ``B``).

    ``logp_*`` and ``mask`` are ``(B, C)`` token tensors, ``advantages`` is ``(B,)`` (one per sequence).
    """
    s = sequence_ratio(logp_new, logp_old, mask)
    a = advantages.detach()
    obj = torch.minimum(s * a, torch.clamp(s, 1.0 - clip_low, 1.0 + clip_high) * a)
    denom = float(len(s)) if normalizer is None else normalizer
    denom = torch.as_tensor(denom, dtype=obj.dtype, device=obj.device)
    return -obj.sum() / denom.clamp_min(1.0)


def gspo_diagnostics(
    logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor,
    clip_low: float, clip_high: float,
) -> Dict[str, float]:
    s = sequence_ratio(logp_new, logp_old, mask).detach()
    a = advantages.detach()
    clipped = torch.clamp(s, 1.0 - clip_low, 1.0 + clip_high) * a < s * a
    return {"mean_seq_ratio": float(s.mean()), "clip_frac": float(clipped.float().mean())}


# ------------------------------------------------------------------------------------------------ trainer
@dataclass
class GSPOConfig:
    # ASSUMPTION: every value in this config. clip_low/clip_high should be checked against the GSPO paper.
    iters: int = 100
    prompts_per_iter: int = 8
    group_size: int = 8
    clip_low: float = 3e-4
    clip_high: float = 4e-4
    adv_eps: float = 1e-6
    n_minibatches: int = 1
    max_new_tokens: int = 64
    max_prompt_len: Optional[int] = None
    temperature: float = 1.0
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    micro_batch_size: Optional[int] = None
    seed: int = 0


def _as_dict(r: RewardOut) -> Dict[str, float]:
    return {"reward": float(r)} if not isinstance(r, dict) else {k: float(v) for k, v in r.items()}


class GSPOTrainer:
    """Group-sampling policy optimisation loop over records ``{"prompt": str, ...}`` scored by ``reward_fn``."""

    def __init__(self, policy: torch.nn.Module, tok: Any, reward_fn: RewardFn, cfg: GSPOConfig):
        self.policy, self.tok, self.reward_fn, self.cfg = policy, tok, reward_fn, cfg
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.opt = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.history: List[Dict[str, float]] = []
        self._it = 0

    def _score(self, records: Sequence[Dict[str, Any]], roll: Rollout) -> Dict[str, torch.Tensor]:
        G = self.cfg.group_size
        outs = [
            _as_dict(self.reward_fn(records[i // G], roll.texts[i], len(roll.completion_ids[i])))
            for i in range(len(roll))
        ]
        keys = set().union(*[o.keys() for o in outs])
        return {k: torch.tensor([o.get(k, 0.0) for o in outs]).view(len(records), G) for k in keys}

    def step(self, records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        cfg, G = self.cfg, self.cfg.group_size
        prompts = [r["prompt"] for r in records for _ in range(G)]
        roll = sample_rollouts(self.policy, self.tok, prompts, cfg.max_new_tokens, cfg.temperature, cfg.max_prompt_len,
                               cfg.micro_batch_size)
        scores = self._score(records, roll)
        rewards = scores["reward"]
        adv = group_advantages(rewards, cfg.adv_eps).reshape(-1)
        logp_old, _ = completion_logprobs(self.policy, roll.prompt_ids, roll.completion_ids, self.pad,
                                          cfg.temperature, False, cfg.micro_batch_size)

        self.policy.train()
        n_prompts = len(records)
        order = list(range(n_prompts))
        per_mb = max(1, -(-n_prompts // cfg.n_minibatches))
        diag = {"mean_seq_ratio": 1.0, "clip_frac": 0.0}
        loss_val, gnorm, n_updates = 0.0, 0.0, 0
        for s in range(0, n_prompts, per_mb):
            prompt_idx = order[s : s + per_mb]
            seq_idx = [p * G + g for p in prompt_idx for g in range(G) if float(adv[p * G + g]) != 0.0]
            if not seq_idx:
                continue
            norm = float(len(prompt_idx) * G)
            self.opt.zero_grad(set_to_none=True)
            mb = cfg.micro_batch_size or len(seq_idx)
            loss_val = 0.0
            for c in range(0, len(seq_idx), mb):
                idx = seq_idx[c : c + mb]
                lp_new, m = completion_logprobs(self.policy, [roll.prompt_ids[i] for i in idx],
                                                [roll.completion_ids[i] for i in idx], self.pad, cfg.temperature, True)
                lp_old = logp_old[idx, : lp_new.size(1)]
                a = adv[idx]
                loss = gspo_loss(lp_new, lp_old, a, m, cfg.clip_low, cfg.clip_high, normalizer=norm)
                loss.backward()
                loss_val += float(loss.detach())
                diag = gspo_diagnostics(lp_new, lp_old, a, m, cfg.clip_low, cfg.clip_high)
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm))
            self.opt.step()
            n_updates += 1
        self._it += 1
        rec: Dict[str, float] = {
            "iter": self._it,
            "mean_reward": float(rewards.mean()),
            "reward_std_within_group": float(rewards.std(dim=-1, correction=0).mean()),
            "frac_groups_with_signal": float((rewards.std(dim=-1, correction=0) > 0).float().mean()),
            "loss": loss_val,
            "grad_norm": gnorm,
            "n_optimizer_steps": float(n_updates),
            "mean_completion_len": float(sum(roll.lengths()) / len(roll)),
            "format_ok_frac": sum(1 for t in roll.texts if "<op>" in t) / len(roll),
            **diag,
        }
        for k, v in scores.items():
            if k != "reward":
                rec[f"mean_{k}"] = float(v.mean())
        self.history.append(rec)
        return rec

    def fit(
        self,
        records: Sequence[Dict[str, Any]],
        iters: Optional[int] = None,
        on_step: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        rng = random.Random(self.cfg.seed)
        set_seed(self.cfg.seed)
        for _ in range(iters if iters is not None else self.cfg.iters):
            batch = rng.sample(list(records), min(self.cfg.prompts_per_iter, len(records)))
            rec = self.step(batch)
            if on_step:
                on_step(rec)
        return self.history

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)
