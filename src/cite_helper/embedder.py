"""Embedding wrapper around multilingual-e5-small with E5 prefixes."""

from __future__ import annotations

import numpy as np

from .schemas import DEFAULT_MODEL_NAME


class Embedder:
    """Caches the SentenceTransformer model for the process lifetime.

    E5 family expects 'query: ' prefix for queries and 'passage: ' for
    documents. Cosine sim is used directly on normalized vectors.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        inputs = [f"passage: {t}" for t in texts]
        embs = self.model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=len(inputs) > 200,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)[0]
