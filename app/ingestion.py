"""
Ingestion: load raw documents (PDF, Markdown, TXT) and split them into
chunks suitable for embedding.

We use a semantic-ish chunking strategy: split on paragraph/section
boundaries first, then greedily pack them up to a target token size with
overlap, instead of blindly cutting every N characters. This avoids
severing a sentence or a table mid-way, which is one of the most common
causes of bad RAG answers.
"""
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir: str) -> List[tuple[str, str]]:
    """Returns list of (source_name, raw_text)."""
    docs = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return docs

    for path in sorted(data_path.glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            docs.append((path.name, _read_pdf(path)))
        elif path.suffix.lower() in {".md", ".txt", ".markdown"}:
            docs.append((path.name, _read_text(path)))
        # Extend here for .html, .docx, etc. via `unstructured` if needed.
    return docs


def _split_into_paragraphs(text: str) -> List[str]:
    # Split on blank lines (paragraph boundaries) and page markers.
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Chunk]:
    """
    Greedy paragraph-packing chunker.

    Packs consecutive paragraphs into a chunk until adding the next one
    would exceed target_chars, then starts a new chunk that repeats the
    tail of the previous one (overlap) so context isn't lost at the seam.
    """
    paragraphs = _split_into_paragraphs(text)
    chunks: List[Chunk] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target_chars:
            chunks.append(
                Chunk(id=str(uuid.uuid4()), text=current.strip(), source=source)
            )
            # carry the tail of the previous chunk forward as overlap
            tail = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = tail + "\n\n" + para
        else:
            current = f"{current}\n\n{para}".strip() if current else para

    if current.strip():
        chunks.append(Chunk(id=str(uuid.uuid4()), text=current.strip(), source=source))

    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata["total_chunks"] = len(chunks)

    return chunks


def build_chunks(data_dir: str) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for source, text in load_documents(data_dir):
        if not text.strip():
            continue
        all_chunks.extend(chunk_text(text, source))
    return all_chunks
