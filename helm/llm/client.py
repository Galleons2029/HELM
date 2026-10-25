"""Chat clients. ``OpenAICompatClient`` talks to any OpenAI-compatible endpoint (e.g. a local vLLM server hosting
Qwen3-8B); ``MockLLM`` is a scripted client for tests and offline runs.

Decoding defaults follow the paper's reproducibility table: temperature 0.1, top-p 0.9, max_tokens 1024.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Union

Message = Dict[str, str]


class ChatClient(Protocol):
    def chat(self, messages: List[Message], **gen) -> str: ...


class OpenAICompatClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-8B",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1024,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai to use OpenAICompatClient") from e
        self.model = model
        self.gen = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self._client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get(api_key_env, "EMPTY"))

    def chat(self, messages: List[Message], **gen) -> str:  # pragma: no cover - needs a live endpoint
        params = {**self.gen, **gen}
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        return resp.choices[0].message.content or ""

    @classmethod
    def from_config(cls, rcfg) -> "OpenAICompatClient":
        return cls(
            model=rcfg.model,
            base_url=rcfg.base_url,
            api_key_env=rcfg.api_key_env,
            temperature=rcfg.temperature,
            top_p=rcfg.top_p,
            max_tokens=rcfg.max_tokens,
        )


class MockLLM:
    """Scripted chat client. ``script`` is either a list of replies (cycled) or ``fn(messages) -> str``."""

    def __init__(self, script: Union[Sequence[str], Callable[[List[Message]], str]]):
        self.script = script
        self.calls: List[List[Message]] = []
        self._i = 0

    def chat(self, messages: List[Message], **gen) -> str:
        self.calls.append(messages)
        if callable(self.script):
            return self.script(messages)
        reply = self.script[self._i % len(self.script)]
        self._i += 1
        return reply
