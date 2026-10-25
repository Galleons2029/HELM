"""Token counting used for CTU and the recall-token cap.

The paper counts tokens "using the backbone tokenizer". Offline we default to a regex word/punctuation tokenizer,
which is a stable proxy; pass ``fn`` (e.g. ``lambda s: len(hf_tokenizer.encode(s))``) to use a real tokenizer.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter:
    def __init__(self, fn: Optional[Callable[[str], int]] = None):
        self._fn = fn

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._fn is not None:
            return int(self._fn(text))
        return len(_TOKEN_RE.findall(text))

    __call__ = count


DEFAULT_COUNTER = TokenCounter()
