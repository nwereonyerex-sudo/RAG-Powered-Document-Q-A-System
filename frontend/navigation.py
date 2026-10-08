"""Page list shared by the two Streamlit entry points."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def run() -> None:
    st.navigation(
        [
            st.Page(
                ROOT / "frontend" / "views" / "document_qa.py",
                title="Document Q&A",
                default=True,
            ),
            st.Page(
                ROOT / "frontend" / "views" / "cv_job_match.py",
                title="CV & Job Match",
            ),
        ]
    ).run()
