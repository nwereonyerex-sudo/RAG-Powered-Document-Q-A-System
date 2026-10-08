"""Shared presentation for the two Streamlit workflows."""

from __future__ import annotations

import html

import streamlit as st

from rag.config import load_settings

KIND_COLOR = {
    "text": "green",
    "markdown": "violet",
    "html": "amber",
    "pdf": "rose",
}

PAGE_CSS = """
<style>
  .stApp { background: radial-gradient(1200px 500px at 10% -10%, #123524 0%, #070b14 42%); }
  .block-container { max-width: 760px; }
  .nk-hero, .nk-hero h1, .nk-hero p, .nk-card p, .nk-card h3 {
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
  }
  header[data-testid="stHeader"] { background: transparent; }
  .stAppDeployButton, #MainMenu, footer { display: none; }
  [data-testid="stChatMessage"] { background: transparent; border: 0; width: 100%; }
  [data-testid="stChatMessageAvatar"],
  [data-testid="chatAvatarIcon-assistant"],
  [data-testid="chatAvatarIcon-user"] { display: none !important; }
  .nk-answer, .nk-card, .nk-user, .nk-hero { box-sizing: border-box; width: 100%; }
  [data-testid="stChatMessage"] pre,
  [data-testid="stChatMessage"] code {
    white-space: pre-wrap !important;
    word-break: break-word;
  }
  .nk-hero {
    margin: 0 0 1rem 0;
    padding: 1.1rem 1.15rem 1rem;
    border-radius: 18px;
    background:
      linear-gradient(135deg, rgba(34,197,94,0.22), rgba(236,72,153,0.12) 42%, rgba(34,211,238,0.12)),
      #101826;
    border: 1px solid rgba(34,197,94,0.45);
    box-shadow: 0 16px 40px rgba(0,0,0,0.28);
  }
  .nk-kicker {
    display: inline-block;
    margin: 0 0 0.4rem 0;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    background: #22c55e;
    color: #052e16;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .nk-hero h1 { margin: 0; color: #f8fafc; font-size: 1.7rem; line-height: 1.15; }
  .nk-hero p { margin: 0.45rem 0 0; color: #cbd5e1; }
  .nk-user, .nk-answer, .nk-card {
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin: 0.35rem 0 0.7rem;
  }
  .nk-user {
    background: linear-gradient(90deg, #f59e0b, #f97316);
    color: #1c1917;
    font-weight: 700;
  }
  .nk-answer {
    background: linear-gradient(180deg, #132033, #0e1726);
    border: 1px solid rgba(34,197,94,0.55);
    box-shadow: inset 4px 0 0 #22c55e;
  }
  .nk-label {
    margin: 0 0 0.35rem;
    color: #4ade80;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .nk-copy {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
    line-height: 1.55;
    color: #f8fafc;
    font-family: "Segoe UI", sans-serif;
  }
  .nk-row-title { margin: 0.2rem 0 0.45rem; color: #e2e8f0; font-size: 0.95rem; font-weight: 800; }
  .nk-card { border: 1px solid transparent; }
  .nk-green { background: #052e16; border-color: #22c55e; }
  .nk-violet { background: #2e1064; border-color: #c084fc; }
  .nk-amber { background: #451a03; border-color: #fbbf24; }
  .nk-rose { background: #4c0519; border-color: #fb7185; }
  .nk-card h3 { margin: 0.35rem 0; color: #fff; font-size: 1rem; overflow-wrap: anywhere; }
  .nk-pill, .nk-page {
    display: inline-block;
    margin-right: 0.35rem;
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
  }
  .nk-pill { background: #22c55e; color: #052e16; }
  .nk-page { background: #0f172a; color: #e2e8f0; }
  .nk-card p {
    margin: 0;
    color: #e2e8f0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.5;
  }
  div[data-testid="stFileUploader"] {
    background: #101826;
    border: 1px dashed rgba(34,197,94,0.55);
    border-radius: 16px;
    padding: 0.4rem 0.6rem 0.2rem;
  }
</style>
"""

STATUS_TONE = {"met": "green", "partial": "amber", "missing": "rose"}
STATUS_LABEL = {"met": "Met", "partial": "Partial", "missing": "Missing"}


def tighten(text: str) -> str:
    """Drop repeated lines so a looping local model cannot fill the screen."""
    kept: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        key = " ".join(line.lower().split())
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return "\n".join(kept).strip()


def source_cards(documents) -> list[dict]:
    cards = []
    for document in documents:
        lines = []
        for line in tighten(document.page_content).splitlines():
            cleaned = line.lstrip("#").strip()
            if cleaned:
                lines.append(cleaned)
        cards.append(
            {
                "source": document.metadata.get("source", "document"),
                "doc_type": document.metadata.get("doc_type", "text"),
                "page": int(document.metadata.get("page", 0) or 0),
                "preview": " ".join(lines)[:420],
            }
        )
    return cards


def render_exchange(role: str, content: str, sources: list[dict] | None = None) -> None:
    with st.chat_message(role):
        if role == "user":
            st.html(f'<div class="nk-user">{html.escape(content)}</div>')
            return
        answer = tighten(content)
        blocks = [
            '<article class="nk-answer">',
            '<p class="nk-label">Answer</p>',
            f'<p class="nk-copy">{html.escape(answer)}</p>',
            "</article>",
        ]
        if sources:
            blocks.append('<p class="nk-row-title">From the files</p>')
            for source in sources:
                kind = str(source.get("doc_type", "text"))
                tone = KIND_COLOR.get(kind, "green")
                preview = " ".join(
                    line.lstrip("#").strip()
                    for line in str(source.get("preview") or "").splitlines()
                    if line.strip()
                )
                blocks.append(
                    f'<article class="nk-card nk-{tone}">'
                    f'<span class="nk-pill">{html.escape(kind.upper())}</span>'
                    f'<span class="nk-page">Page {int(source.get("page", 0)) + 1}</span>'
                    f"<h3>{html.escape(str(source.get('source', 'document')))}</h3>"
                    f"<p>{html.escape(preview)}</p>"
                    "</article>"
                )
        st.html("".join(blocks))


def hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <section class="nk-hero">
          <p class="nk-kicker">{html.escape(kicker)}</p>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def retrieval_sidebar(prefix: str):
    with st.sidebar:
        st.header("Retrieval")
        st.number_input(
            "Chunk size",
            min_value=200,
            max_value=4000,
            value=1000,
            step=100,
            key=f"{prefix}_chunk_size",
        )
        st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=1000,
            value=200,
            step=50,
            key=f"{prefix}_chunk_overlap",
        )
        st.number_input("Top k", min_value=1, max_value=10, value=4, step=1, key=f"{prefix}_top_k")
        st.selectbox("Search", options=["mmr", "similarity"], key=f"{prefix}_search_type")
        st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key=f"{prefix}_score_threshold",
        )
        st.selectbox("LLM", options=["auto", "openai", "local"], key=f"{prefix}_llm_provider")
    return load_settings(
        chunk_size=int(st.session_state[f"{prefix}_chunk_size"]),
        chunk_overlap=int(st.session_state[f"{prefix}_chunk_overlap"]),
        top_k=int(st.session_state[f"{prefix}_top_k"]),
        search_type=st.session_state[f"{prefix}_search_type"],
        score_threshold=float(st.session_state[f"{prefix}_score_threshold"]),
        llm_provider=st.session_state[f"{prefix}_llm_provider"],
    )


def render_match(match) -> None:
    tone = STATUS_TONE.get(match.status, "amber")
    label = STATUS_LABEL.get(match.status, match.status)
    evidence = match.cv_evidence or "No CV line supports this requirement."
    st.html(
        "".join(
            [
                f'<article class="nk-card nk-{tone}">',
                f'<span class="nk-pill">{html.escape(label)}</span>',
                f"<h3>Requirement: {html.escape(match.requirement)}</h3>",
                f"<p><strong>CV evidence:</strong> “{html.escape(evidence)}”</p>",
                "<p><strong>Suggested improvement</strong></p>",
                f"<p>{html.escape(match.suggestion)}</p>",
                "</article>",
            ]
        )
    )
