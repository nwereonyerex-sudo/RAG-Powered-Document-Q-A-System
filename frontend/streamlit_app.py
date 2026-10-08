"""Two separate RAG workflows: document Q&A, and CV to job matching."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from frontend.navigation import run

st.set_page_config(page_title="RAG Document System", layout="centered")
run()
