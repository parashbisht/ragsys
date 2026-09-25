# Production-Grade RAG System

A retrieval-augmented generation (RAG) system built to demonstrate real
engineering depth — not just "LangChain + a PDF". Covers ingestion, hybrid
retrieval (BM25 + vector), reranking, citation-aware generation, and an
evaluation harness.

## Why this exists

Most RAG portfolio projects stop at "embed chunks → cosine similarity → ask
LLM". This one goes further, because that's what separates a junior
candidate from someone who understands production RAG:

- **Hybrid retrieval** (keyword + semantic) instead of vector-only search
- **Reranking** to fix the common problem of relevant-looking-but-wrong chunks
- **Citations** in every answer, tied back to source chunks
- **Evaluation harness** with faithfulness/relevance scoring, not just vibes
- **A working API + UI**, not just a notebook

## Architecture

```
   ┌─────────────┐
   │  Documents  │  (PDF, MD, TXT, HTML)
   └──────┬──────┘
          │ ingestion.py — parse + semantic chunking
          ▼
   ┌─────────────┐
   │  Chunks +   │
   │  Metadata   │
   └──────┬──────┘
          │ embeddings.py — sentence-transformers or OpenAI
          ▼
   ┌─────────────┐        ┌──────────────┐
   │ Vector Store│        │  BM25 Index  │
   │   (FAISS)   │        │  (keyword)   │
   └──────┬──────┘        └───────┬──────┘
          │                       │
          └───────────┬───────────┘
                       ▼
              retriever.py — hybrid merge + rerank
                       ▼
              generation.py — LLM answer + citations
                       ▼
                 FastAPI (app/main.py)
                       ▼
              Streamlit UI (frontend/streamlit_app.py)
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY

# 3. Add your documents
# Drop .pdf, .md, or .txt files into data/sample/

# 4. Build the index
python scripts/ingest.py

# 5. Run the API
uvicorn app.main:app --reload --port 8000

# 6. (Optional) Run the UI
streamlit run frontend/streamlit_app.py
```

Then visit `http://localhost:8000/docs` for the interactive API,
or the Streamlit URL it prints for the chat UI.

## Evaluate it

```bash
python scripts/evaluate.py
```

This runs a small Q&A eval set (edit `scripts/evaluate.py` to add your own
questions + expected answers) and reports faithfulness and answer-relevance
scores using an LLM-as-judge approach. Log this in your README/portfolio —
recruiters notice when you can *quantify* how well your system performs,
not just show a demo.

## Deploying it for real users

See the "Next steps" section below — this scaffold is designed to be
deployed on Render/Railway (backend) + Streamlit Community Cloud or Vercel
(frontend) with minimal changes. Swap FAISS for a hosted vector DB
(Pinecone/Qdrant Cloud) once you have real concurrent users.

## Project structure

```
app/
  main.py          FastAPI app + endpoints
  config.py        Settings (env vars, model names)
  ingestion.py      Document loading + chunking
  embeddings.py     Embedding model wrapper
  vectorstore.py    FAISS index build/load/search
  retriever.py      Hybrid (BM25 + vector) retrieval + rerank
  generation.py     LLM call with citation-aware prompting
  schemas.py        Pydantic request/response models
scripts/
  ingest.py         CLI: build the index from data/sample/
  evaluate.py       CLI: run eval harness, print scores
frontend/
  streamlit_app.py  Minimal chat UI
tests/
  test_retriever.py Basic retrieval sanity tests
```

## Next steps (to make this "real users can use it" grade)

1. **Auth + rate limiting** — add API-key auth (`app/auth.py`) and a
   rate limiter (e.g. `slowapi`) to control cost per user.
2. **Persistent chat history** — add Postgres/Supabase for session storage.
3. **Swap FAISS → hosted vector DB** (Pinecone/Qdrant Cloud) so the index
   survives redeploys and scales past one machine.
4. **Add a feedback endpoint** (`/feedback`) logging thumbs up/down per
   answer — use this in interviews to talk about iteration.
5. **Deploy**: backend to Render/Railway, UI to Streamlit Cloud, or wrap
   both in one Docker container (see `docker/Dockerfile`) and deploy
   anywhere.
6. **Observability**: log latency, token usage, and retrieval-failure rate
   per request (a simple `app/logging_middleware.py` is enough to start).
