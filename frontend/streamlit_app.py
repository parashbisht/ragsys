"""
Minimal chat UI for the RAG API.

Run with:
    streamlit run frontend/streamlit_app.py

Assumes the FastAPI backend is running at http://localhost:8000
(set API_URL env var to point elsewhere, e.g. your deployed backend).
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Chat with your docs", page_icon="💬")
st.title("💬 Chat with your docs")
st.caption("A production-grade RAG system — hybrid retrieval + reranking + citations.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander(f"📄 {len(sources)} source(s)"):
                        for s in sources:
                            st.markdown(f"**[{s['index']}] {s['source']}**")
                            st.caption(s["text"])

                col1, col2 = st.columns(2)
                if col1.button("👍 Helpful", key=f"up_{len(st.session_state.history)}"):
                    requests.post(
                        f"{API_URL}/feedback",
                        json={"question": question, "answer": answer, "helpful": True},
                    )
                if col2.button("👎 Not helpful", key=f"down_{len(st.session_state.history)}"):
                    requests.post(
                        f"{API_URL}/feedback",
                        json={"question": question, "answer": answer, "helpful": False},
                    )

                st.session_state.history.append({"role": "assistant", "content": answer})
            except requests.exceptions.RequestException as e:
                st.error(f"Couldn't reach the API at {API_URL}. Is it running? ({e})")
