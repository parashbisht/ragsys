"""
Thin wrapper around FAISS for local vector storage. For real multi-user
deployments, swap this for a hosted vector DB (Pinecone/Qdrant Cloud) —
the build_index/search interface below is designed so that swap only
touches this file.
"""
import json
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np

from app.embeddings import embed_texts
from app.ingestion import Chunk


class VectorStore:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)
        dim = vectors.shape[1]

        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")
        query_vector = np.expand_dims(query_vector, axis=0)
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
