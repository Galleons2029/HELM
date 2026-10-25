"""LLM-driven and learned controllers (same ``decide`` interface as the heuristic controllers).

``LLMController``  prompts any chat client (prompted 5-shot-style baseline, or a served learned controller) with
                   ``render_state`` and parses ``<op>`` lines. If the output does not parse it falls back to the
                   heuristic policy, and counts the fallback so it shows up in the diagnostics.
``HFController``   loads a local HF causal-LM checkpoint (e.g. the output of scripts/train_controller_*.py) and
                   generates ops greedily. Heavy dependencies (torch/transformers) are imported lazily.
"""

from __future__ import annotations

from typing import List, Optional

from ..llm.client import ChatClient
from .base import ControllerState, MemOp, OpKind, parse_ops, render_state
from .heuristic import HeuristicController

SYSTEM_PROMPT = (
    "You manage an agent's external memory. Given the current state, output the memory operations to run now, one per "
    "line: <op>READ</op><q>query</q>, <op>WRITE</op>, <op>CONSOLIDATE</op>, <op>PRUNE</op> or <op>NOOP</op>. "
    "Write only observations worth keeping (including failed tool traces, which are kept as evidence). "
    "Always answer a recall request with READ. Do not consolidate or write redundantly."
)


class LLMController:
    name = "llm"

    def __init__(self, client: ChatClient, fallback: Optional[HeuristicController] = None):
        self.client = client
        self.fallback = fallback or HeuristicController()
        self.n_calls = 0
        self.n_fallbacks = 0

    def decide(self, s: ControllerState) -> List[MemOp]:
        self.n_calls += 1
        reply = self.client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": render_state(s)}]
        )
        if "<op>" not in reply:  # unparseable output: defer to the heuristic policy and count it
            self.n_fallbacks += 1
            return self.fallback.decide(s)
        ops = parse_ops(reply)
        if s.recall_request and not any(o.kind == OpKind.READ for o in ops):
            ops.append(MemOp(OpKind.READ, s.recall_request))
        for o in ops:
            if o.kind == OpKind.READ and not o.query:
                o.query = s.recall_request
        return ops


class HFController:  # pragma: no cover - needs weights; exercised by the training smoke tests instead
    name = "learned"

    def __init__(self, checkpoint: str, device: Optional[str] = None, max_new_tokens: int = 64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def decide(self, s: ControllerState) -> List[MemOp]:
        prompt = SYSTEM_PROMPT + "\n\n" + render_state(s) + "\n"
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        ops = parse_ops(text)
        for o in ops:
            if o.kind == OpKind.READ and not o.query:
                o.query = s.recall_request
        return ops
