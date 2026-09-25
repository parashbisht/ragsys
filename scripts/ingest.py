"""
Build (or rebuild) the vector index from documents in DATA_DIR.

Usage:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion import build_chunks
from app.vectorstore import VectorStore


def main() -> None:
    print(f"Loading documents from {settings.data_dir} ...")
    chunks = build_chunks(settings.data_dir)

    if not chunks:
        print(
            f"No documents found in {settings.data_dir}. "
            "Add .pdf, .md, or .txt files there and re-run."
        )
        return

    print(f"Built {len(chunks)} chunks from source documents.")
    print("Embedding and building FAISS index (this may take a minute)...")

    store = VectorStore(settings.index_dir)
    store.build(chunks)
    store.save()

    print(f"Index saved to {settings.index_dir}/")
    print("You can now run: uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
