"""
Generation: takes the retrieved chunks and produces an answer that cites
its sources. We number the chunks and instruct the model to reference
them inline (e.g. "[1]"), then return the numbered source list alongside
the answer so the UI can render clickable citations.
"""
from typing import List

from app.config import settings
from app.ingestion import Chunk

SYSTEM_PROMPT = """You are a document-grounded assistant. Use ONLY information explicitly \
stated in the provided source excerpts. Never use your general knowledge, \
training data, assumptions, or outside information to answer or fill gaps. If \
the excerpts are insufficient, irrelevant, or do not explicitly contain the \
answer, clearly say: "The provided documents do not contain the answer." Do not \
guess or infer an answer from outside the excerpts.

When the excerpts do support an answer, cite sources inline using [1], [2], etc., \
matching the source numbers below. Be concise and direct.
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


def _call_gemini(prompt: str) -> str:
    import time
    from google import genai
    from google.genai import errors

    client = genai.Client(api_key=settings.google_api_key)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            )

            return response.text

        except errors.ServerError as e:
            if getattr(e, "status_code", None) == 503:
                if attempt < 2:
                    wait_time = 2 ** attempt

                    print(
                        f"Gemini temporarily unavailable. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)
                    continue

            raise


def generate_answer(query: str, chunks: List[Chunk]) -> dict:
    context = _build_context(chunks)
    prompt = f"Sources:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

    if settings.llm_provider == "anthropic":
        answer = _call_anthropic(prompt)
    elif settings.llm_provider == "openai":
        answer = _call_openai(prompt)
    elif settings.llm_provider == "gemini":
        answer = _call_gemini(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")

    return {
        "answer": answer,
        "sources": [
            {"index": i + 1, "source": c.source, "text": c.text[:300]}
            for i, c in enumerate(chunks)
        ],
    }
