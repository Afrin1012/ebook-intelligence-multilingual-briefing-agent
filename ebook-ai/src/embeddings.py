"""Text embedding utilities."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Embed a batch of strings; vectors are L2-normalized for cosine similarity."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    model = get_embedding_model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query], model_name=model_name)[0]
