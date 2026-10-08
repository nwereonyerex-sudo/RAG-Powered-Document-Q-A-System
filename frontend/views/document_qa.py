"""Document question answering. CV matching lives on its own page."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from frontend.ui import hero, render_exchange, retrieval_sidebar, source_cards
from rag.chain import ask, describe_document, wants_document_review
from rag.ingest import SUPPORTED
from rag.store import index_directory, index_paths

hero(
    "Document Q&A",
    "Ask a document a question",
    "Upload a paper, filing, report, or book. The answer stays inside the retrieved passages, with the file and page beside it.",
)
settings = retrieval_sidebar("qa")
st.sidebar.caption("CV and job-description matching is on the CV & Job Match page.")

uploads = st.file_uploader(
    "Upload PDF, TXT, Markdown, or HTML",
    type=["pdf", "txt", "md", "html", "htm"],
    accept_multiple_files=True,
    key="qa_uploads",
)
include_samples = st.checkbox("Also index the bundled samples", value=False, key="qa_samples")
limit_to_latest = st.checkbox(
    "Search only the files indexed in this session",
    value=False,
    key="qa_limit",
)

if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []
if "qa_indexed_names" not in st.session_state:
    st.session_state.qa_indexed_names = []


def index_selection() -> int:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for upload in uploads or []:
        destination = settings.raw_dir / Path(upload.name).name
        destination.write_bytes(upload.getvalue())
        saved.append(destination)
    if not saved and not include_samples:
        raise ValueError("Choose a file first, then index it.")
    count = index_paths(saved, settings) if saved else 0
    names = [path.name for path in saved]
    if include_samples:
        count += index_directory(ROOT / "samples", settings)
        names.extend(
            path.name
            for path in (ROOT / "samples").iterdir()
            if path.suffix.lower() in SUPPORTED
        )
    st.session_state.qa_indexed_names = names
    st.session_state.qa_upload_signature = (
        tuple((upload.name, upload.size) for upload in (uploads or [])),
        include_samples,
    )
    return count


upload_signature = (
    tuple((upload.name, upload.size) for upload in (uploads or [])),
    include_samples,
)
if upload_signature[0] and upload_signature != st.session_state.get("qa_upload_signature"):
    try:
        count = index_selection()
        st.success(f"Indexed {', '.join(st.session_state.qa_indexed_names)} ({count} chunks).")
    except Exception as exc:
        st.error(str(exc))

if st.button("Index documents", key="qa_index"):
    try:
        count = index_selection()
        st.success(f"Stored {count} chunks in `{settings.collection_name}`.")
    except Exception as exc:
        st.error(str(exc))

source_filter = None
names = st.session_state.qa_indexed_names
if limit_to_latest and len(names) == 1:
    source_filter = names[0]
elif limit_to_latest and len(names) > 1:
    source_filter = st.selectbox("Limit search to", names, key="qa_source_limit")

for message in st.session_state.qa_messages:
    render_exchange(message["role"], message["content"], message.get("sources"))

question = st.chat_input("Ask a question about the indexed documents")
if question:
    st.session_state.qa_messages.append({"role": "user", "content": question})
    render_exchange("user", question)
    try:
        if wants_document_review(question) and source_filter:
            result = describe_document(settings, source_filter)
        else:
            result = ask(question, settings, source_filter=source_filter)
        cards = source_cards(result.sources)
        st.session_state.qa_messages.append(
            {"role": "assistant", "content": result.text, "sources": cards}
        )
        render_exchange("assistant", result.text, cards)
    except Exception as exc:
        st.error(str(exc))
