"""
Generation: takes the retrieved chunks and produces an answer that cites
its sources. We number the chunks and instruct the model to reference
them inline (e.g. "[1]"), then return the numbered source list alongside
the answer so the UI can render clickable citations.
"""
from typing import List

from app.config import settings
from app.ingestion import Chunk

SYSTEM_PROMPT = """You are a careful assistant that answers questions using ONLY the \
provided source excerpts. Rules:

1. Answer using only information in the sources below. If the sources don't \
contain the answer, say so plainly — never fabricate.
2. Cite sources inline using [1], [2], etc., matching the source numbers below.
3. Be concise and direct.
"""


def _build_context(chunks: List[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c.source})\n{c.text}")
    return "\n\n".join(parts)


def _call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def generate_answer(query: str, chunks: List[Chunk]) -> dict:
    context = _build_context(chunks)
    prompt = f"Sources:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

    if settings.llm_provider == "anthropic":
        answer = _call_anthropic(prompt)
    elif settings.llm_provider == "openai":
        answer = _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")

    return {
        "answer": answer,
        "sources": [
            {"index": i + 1, "source": c.source, "text": c.text[:300]}
            for i, c in enumerate(chunks)
        ],
    }
