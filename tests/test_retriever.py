"""
Basic sanity tests. Run with: pytest tests/

These aren't exhaustive — they check that the pipeline's core pieces
(chunking, retrieval ranking) behave sensibly, which is enough to catch
regressions as you extend the project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import chunk_text
from app.vectorstore import VectorStore
from app.retriever import HybridRetriever


def test_chunking_produces_nonempty_chunks():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three." * 20
    chunks = chunk_text(text, source="test.txt", target_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)


def test_chunking_preserves_content():
    text = "Unique marker ABC123 appears here.\n\nAnd some more text follows after it."
    chunks = chunk_text(text, source="test.txt")
    combined = " ".join(c.text for c in chunks)
    assert "ABC123" in combined


def test_hybrid_retriever_ranks_relevant_chunk_higher():
    from app.ingestion import Chunk

    chunks = [
        Chunk(id="1", text="The capital of France is Paris.", source="a.txt"),
        Chunk(id="2", text="Bananas are a good source of potassium.", source="b.txt"),
        Chunk(id="3", text="Paris is known for the Eiffel Tower.", source="c.txt"),
    ]
    store = VectorStore(index_dir="/tmp/test_index")
    store.build(chunks)

    retriever = HybridRetriever(store)
    results = retriever.retrieve("What is the capital of France?", top_k=3)

    assert len(results) > 0
    # The banana chunk should not be the top result for a France question.
    assert results[0].text != "Bananas are a good source of potassium."
