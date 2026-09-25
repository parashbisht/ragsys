"""
Hybrid retrieval: combine BM25 (keyword) and vector (semantic) search,
then rerank the merged candidate pool with a cross-encoder.

Why hybrid: pure vector search misses exact keyword/ID/acronym matches
(e.g. "error code E204" or a product SKU) that BM25 catches trivially.
Pure BM25 misses paraphrases and synonyms that vector search catches.
Combining both and reranking the union is the standard production
pattern — this is usually the single biggest quality jump over a
vector-only RAG demo.
"""
from functools import lru_cache
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings
from app.embeddings import embed_query
from app.ingestion import Chunk
from app.vectorstore import VectorStore


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model)


class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.chunks: List[Chunk] = vector_store.chunks
        self._bm25 = self._build_bm25(self.chunks)

    @staticmethod
    def _build_bm25(chunks: List[Chunk]) -> BM25Okapi:
        tokenized = [c.text.lower().split() for c in chunks]
        return BM25Okapi(tokenized)

    def _bm25_scores(self, query: str) -> np.ndarray:
        tokenized_query = query.lower().split()
        scores = np.array(self._bm25.get_scores(tokenized_query))
        # normalize to [0, 1] so it's comparable with cosine similarity
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def retrieve(self, query: str, top_k: int | None = None) -> List[Chunk]:
        top_k = top_k or settings.top_k_retrieve

        # --- Vector search ---
        query_vec = embed_query(query)
        vector_results = self.vector_store.search(query_vec, top_k=top_k)
        vector_scores = {id(c): score for c, score in vector_results}

        # --- BM25 search ---
        bm25_scores_all = self._bm25_scores(query)
        top_bm25_idx = np.argsort(bm25_scores_all)[::-1][:top_k]

        # --- Merge candidate pool ---
        candidates: dict[int, Chunk] = {}
        for c, _ in vector_results:
            candidates[id(c)] = c
        for idx in top_bm25_idx:
            c = self.chunks[idx]
            candidates[id(c)] = c

        # --- Fuse scores (weighted sum) ---
        alpha = settings.hybrid_alpha
        fused: List[tuple[Chunk, float]] = []
        for c in candidates.values():
            v_score = vector_scores.get(id(c), 0.0)
            b_idx = self.chunks.index(c) if c in self.chunks else None
            b_score = bm25_scores_all[b_idx] if b_idx is not None else 0.0
            fused_score = alpha * v_score + (1 - alpha) * b_score
            fused.append((c, fused_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        candidate_chunks = [c for c, _ in fused[:top_k]]

        # --- Rerank with cross-encoder ---
        return self._rerank(query, candidate_chunks)

    def _rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []
        reranker = _get_reranker()
        pairs = [[query, c.text] for c in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[: settings.top_k_final]]
