"""
Embedding wrapper. Runs locally via sentence-transformers by default so
the project works with zero API cost during development. Swap in an
OpenAI/Voyage embedding call here later if you need higher quality at
scale — the interface stays the same.
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
