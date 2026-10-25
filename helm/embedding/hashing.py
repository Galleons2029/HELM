"""Deterministic offline embedder (signed feature hashing of word uni/bi-grams).

Why this exists: the repo must run with no downloads and no GPU. This is NOT a stand-in for Qwen3-Embedding-4B's
quality; it is a fixed, reproducible encoder with the one property the ablations need -- it is *status-blind* unless
``tool_aware`` is on. With ``tool_aware=True`` it additionally exposes the structured tool-return fields (return code)
as a hashed feature channel, and instruction-conditioned queries are pulled toward successful traces. That channel is a
proxy for the paper's tool-aware fine-tuning with epistemic hard negatives (Sec. "Tool-Aware Embedding Optimization"),
and turning it off is the proxy for the "-Emb" ablation.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import l2_normalize

_WORD = re.compile(r"[a-z0-9]+")
_CODE = re.compile(r"""["']?code["']?\s*[:=]\s*(\d{3})""")
# Fields that appear in (almost) every tool trace carry little retrieval signal; down-weight like an IDF would.
_BOILERPLATE = {"entity", "attr", "value", "code", "version", "status", "tool", "step", "ts", "error", "msg"}


class HashingEmbedder:
    def __init__(self, dim: int = 512, tool_aware: bool = True, seed: int = 0, struct_weight: float = 0.8):
        self.dim = dim
        self.tool_aware = tool_aware
        self.struct_weight = struct_weight
        self._salt = f"s{seed}|"
        self._cache: Dict[str, Tuple[int, float]] = {}

    # ------------------------------------------------------------------
    def _bucket(self, feat: str) -> Tuple[int, float]:
        hit = self._cache.get(feat)
        if hit is None:
            h = zlib.crc32((self._salt + feat).encode("utf-8"))
            hit = (h % self.dim, 1.0 if (h >> 31) & 1 else -1.0)
            self._cache[feat] = hit
        return hit

    def _features(self, text: str) -> List[Tuple[str, float]]:
        toks = _WORD.findall(text.lower())
        feats: Dict[str, float] = {}
        for t in toks:
            w = 0.25 if t in _BOILERPLATE else 1.0
            feats["u:" + t] = feats.get("u:" + t, 0.0) + w
        for a, b in zip(toks, toks[1:]):
            if a in _BOILERPLATE or b in _BOILERPLATE:
                continue
            feats["b:" + a + "_" + b] = feats.get("b:" + a + "_" + b, 0.0) + 1.0
        # sublinear tf
        return [(f, 1.0 + math.log(c) if c > 0 else 0.0) for f, c in feats.items()]

    def _struct(self, text: str, is_query: bool) -> List[Tuple[str, float]]:
        if not self.tool_aware:
            return []
        if is_query:
            return [("s:ok", self.struct_weight * 2.0)]
        m = _CODE.search(text)
        if not m:
            return []
        code = int(m.group(1))
        return [("s:ok" if 200 <= code < 300 else "s:bad", self.struct_weight * 2.0)]

    def encode(self, texts: Sequence[str], instruction: Optional[str] = None) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat, w in self._features(text) + self._struct(text, instruction is not None):
                idx, sign = self._bucket(feat)
                out[i, idx] += sign * w
        return l2_normalize(out)
