"""Upload documents and ask grounded questions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from rag.chain import ask
from rag.config import load_settings
from rag.ingest import SUPPORTED
from rag.store import index_directory, index_paths

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("RAG-Powered Document Q&A")
st.caption(
    "Upload a paper, filing, report, or book. Ask in plain English. "
    "The answer is limited to retrieved chunks, with the file and page beside it."
)


def _settings():
    return load_settings(
        chunk_size=int(st.session_state.chunk_size),
        chunk_overlap=int(st.session_state.chunk_overlap),
        top_k=int(st.session_state.top_k),
        search_type=st.session_state.search_type,
        score_threshold=float(st.session_state.score_threshold),
        llm_provider=st.session_state.llm_provider,
    )


with st.sidebar:
    st.header("Retrieval")
    st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100, key="chunk_size")
    st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50, key="chunk_overlap")
    st.number_input("Top k", min_value=1, max_value=10, value=4, step=1, key="top_k")
    st.selectbox("Search", options=["mmr", "similarity"], key="search_type")
    st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="score_threshold")
    st.selectbox("LLM", options=["auto", "openai", "local"], key="llm_provider")
    st.caption("auto uses OpenAI when OPENAI_API_KEY is set, otherwise a local Hugging Face model. Threshold applies to similarity search.")

settings = _settings()
st.sidebar.code(settings.collection_name, language="text")

uploads = st.file_uploader(
    "Upload PDF, TXT, Markdown, or HTML",
    type=["pdf", "txt", "md", "html", "htm"],
    accept_multiple_files=True,
)
include_samples = st.checkbox("Also index the bundled samples", value=True)
limit_to_latest = st.checkbox("Search only the files indexed in this session", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_names" not in st.session_state:
    st.session_state.indexed_names = []

if st.button("Index documents", type="primary"):
    try:
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for upload in uploads or []:
            destination = settings.raw_dir / Path(upload.name).name
            destination.write_bytes(upload.getvalue())
            saved.append(destination)
        count = index_paths(saved, settings) if saved else 0
        if include_samples:
            count += index_directory(ROOT / "samples", settings)
        names = [path.name for path in saved]
        if include_samples:
            names.extend(path.name for path in (ROOT / "samples").iterdir() if path.suffix.lower() in SUPPORTED)
        st.session_state.indexed_names = names
        st.success(f"Stored {count} chunks in `{settings.collection_name}`.")
    except Exception as exc:
        st.error(str(exc))

source_filter = None
if limit_to_latest and len(st.session_state.indexed_names) == 1:
    source_filter = st.session_state.indexed_names[0]
elif limit_to_latest and len(st.session_state.indexed_names) > 1:
    source_filter = st.selectbox("Limit search to", st.session_state.indexed_names)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the indexed documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = ask(question, settings, source_filter=source_filter)
            st.markdown(result.text)
            with st.expander("Retrieved chunks"):
                for document in result.sources:
                    page = int(document.metadata.get("page", 0)) + 1
                    st.markdown(
                        f"**{document.metadata.get('source')}** · page {page} · {document.metadata.get('doc_type')}"
                    )
                    st.text(document.page_content)
            st.session_state.messages.append({"role": "assistant", "content": result.text})
        except Exception as exc:
            st.error(str(exc))
