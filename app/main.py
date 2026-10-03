"""
FastAPI app exposing the RAG pipeline.

Run with:
    uvicorn app.main:app --reload --port 8000

Then visit http://localhost:8000/docs for the interactive API.
"""
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.generation import generate_answer
from app.retriever import HybridRetriever
from app.schemas import FeedbackRequest, QueryRequest, QueryResponse
from app.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

app = FastAPI(title="Production RAG API", version="0.1.0")
allowed_origins = [
    origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_vector_store: VectorStore | None = None
_retriever: HybridRetriever | None = None


@app.on_event("startup")
def load_index() -> None:
    global _vector_store, _retriever
    index_path = Path(settings.index_dir) / "faiss.index"
    if not index_path.exists():
        logger.warning(
            "No index found at %s. Run `python scripts/ingest.py` first.",
            settings.index_dir,
        )
        return
    _vector_store = VectorStore(settings.index_dir)
    _vector_store.load()
    _retriever = HybridRetriever(_vector_store)
    logger.info("Loaded index with %d chunks.", len(_vector_store.chunks))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_loaded": _retriever is not None,
        "num_chunks": len(_vector_store.chunks) if _vector_store else 0,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run scripts/ingest.py first, then restart the API.",
        )

    start = time.time()
    chunks = _retriever.retrieve(request.question, top_k=request.top_k)
    if not chunks:
        return QueryResponse(
            answer="I couldn't find anything relevant in the documents to answer that.",
            sources=[],
        )

    result = generate_answer(request.question, chunks)
    elapsed = time.time() - start
    logger.info(
        "query='%s' chunks=%d latency=%.2fs",
        request.question,
        len(chunks),
        elapsed,
    )
    return QueryResponse(**result)


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    # Minimal stub: append to a local log file. Swap for a real DB in
    # production so you can analyze which answers users found unhelpful.
    with open("feedback.log", "a", encoding="utf-8") as f:
        f.write(request.model_dump_json() + "\n")
    return {"status": "recorded"}
