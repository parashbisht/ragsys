"""
Evaluation harness. Runs a set of test questions through the pipeline and
scores each answer on two axes using an LLM-as-judge:

  - Faithfulness: is the answer actually supported by the retrieved
    sources, or does it hallucinate beyond them?
  - Relevance: does the answer actually address the question asked?

This is a lightweight stand-in for tools like RAGAS — swap in RAGAS
directly if you want more rigorous, published metrics for your portfolio
writeup.

Usage:
    python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.generation import generate_answer
from app.retriever import HybridRetriever
from app.vectorstore import VectorStore

# --- Edit this with real questions about YOUR documents ---
EVAL_SET = [
    {
        "question": "What is this document about?",
        "expected_topic": "a high-level summary of the ingested documents",
    },
    # Add more: {"question": "...", "expected_topic": "..."},
]

JUDGE_PROMPT = """You are grading a RAG system's answer.

Question: {question}

Retrieved sources:
{sources}

Generated answer:
{answer}

Score the answer from 1-5 on two dimensions:
1. Faithfulness: Is every claim in the answer supported by the sources? \
(5 = fully supported, 1 = mostly fabricated)
2. Relevance: Does the answer address the question? \
(5 = fully addresses it, 1 = off-topic)

Respond ONLY with JSON: {{"faithfulness": <int>, "relevance": <int>, "reasoning": "<one sentence>"}}
"""


def judge(question: str, sources_text: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, sources=sources_text, answer=answer)

    if settings.llm_provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
    elif settings.llm_provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        raw = response.choices[0].message.content
    elif settings.llm_provider == "gemini":
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        raw = response.text
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")

    raw = raw.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevance": None, "reasoning": f"unparseable: {raw}"}


def main() -> None:
    index_path = Path(settings.index_dir) / "faiss.index"
    if not index_path.exists():
        print("No index found. Run scripts/ingest.py first.")
        return

    store = VectorStore(settings.index_dir)
    store.load()
    retriever = HybridRetriever(store)

    results = []
    for item in EVAL_SET:
        question = item["question"]
        chunks = retriever.retrieve(question)
        if not chunks:
            print(f"[SKIP] No chunks retrieved for: {question}")
            continue

        result = generate_answer(question, chunks)
        sources_text = "\n\n".join(c.text for c in chunks)
        scores = judge(question, sources_text, result["answer"])

        results.append({"question": question, **scores})
        print(f"\nQ: {question}")
        print(f"A: {result['answer'][:200]}...")
        print(f"Scores: {scores}")

    if results:
        valid = [r for r in results if r["faithfulness"] is not None]
        if valid:
            avg_faith = sum(r["faithfulness"] for r in valid) / len(valid)
            avg_rel = sum(r["relevance"] for r in valid) / len(valid)
            print(f"\n=== Average faithfulness: {avg_faith:.2f}/5 ===")
            print(f"=== Average relevance:    {avg_rel:.2f}/5 ===")


if __name__ == "__main__":
    main()
