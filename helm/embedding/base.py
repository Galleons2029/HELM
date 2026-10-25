"""Embedder interface."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str], instruction: Optional[str] = None) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised embeddings.

        ``instruction`` specifies the retrieval granularity (e.g. evidence span vs. high-level plan), mirroring the
        instruction-aware contrastive tuning of the paper.
        """
        ...


def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, eps)).astype(np.float32)


def build_embedder(cfg) -> Embedder:
    """Factory driven by ``EmbeddingConfig``."""
    if cfg.backend == "hashing":
        from .hashing import HashingEmbedder

        return HashingEmbedder(dim=cfg.dim, tool_aware=cfg.tool_aware, seed=cfg.seed)
    if cfg.backend == "sentence_transformers":
        from .st_backend import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(cfg.model_name)
    raise ValueError(f"unknown embedding backend: {cfg.backend}")
