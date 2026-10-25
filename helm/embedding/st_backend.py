"""sentence-transformers / Qwen3-Embedding backend (optional dependency; NOT exercised in the offline tests).

Usage (needs ``pip install sentence-transformers`` and enough memory for the model; Qwen3-Embedding-4B is ~8 GB in
bf16 and does not fit comfortably on a 16 GB laptop):

    embedder = SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-4B")           # off-the-shelf
    embedder = SentenceTransformerEmbedder("outputs/embedding_ft/final")        # after scripts/train_embedding.py
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import l2_normalize


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: Optional[str] = None, max_seq_length: int = 8192):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install sentence-transformers to use the sentence_transformers backend") from e
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], instruction: Optional[str] = None) -> np.ndarray:
        # Qwen3-Embedding convention: queries are wrapped as "Instruct: <task>\nQuery:<text>"; documents are raw.
        if instruction is not None:
            texts = [f"Instruct: {instruction}\nQuery:{t}" for t in texts]
        vecs = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        return l2_normalize(np.asarray(vecs, dtype=np.float32))
