"""Launch the two-page app. `streamlit run frontend/streamlit_app.py` does the same."""

import streamlit as st

from frontend.navigation import run

st.set_page_config(page_title="RAG Document System", layout="centered")
run()
