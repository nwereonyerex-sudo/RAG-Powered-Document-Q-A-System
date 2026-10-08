"""Match a CV to a job description without using the document Q&A chat."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from backend.services.document_parser import index_file
from backend.services.recommendation_engine import analyze_sources
from frontend.ui import hero, render_match, retrieval_sidebar
from rag.chain import review_cv_for_role
from rag.store import chunks_for_source

hero(
    "CV & Job Match",
    "Check each requirement against the CV",
    "Upload a CV and a job description. Each requirement is marked met, partial, or missing. A partial match quotes the CV and does not add a skill the file does not show.",
)
settings = retrieval_sidebar("cv")
st.sidebar.caption("Questions about a paper, filing, or report stay on Document Q&A.")

cv_upload = st.file_uploader(
    "Your CV",
    type=["pdf", "txt", "md", "html", "htm"],
    accept_multiple_files=False,
    key="cv_match_file",
)
jd_upload = st.file_uploader(
    "Job description",
    type=["pdf", "txt", "md", "html", "htm"],
    accept_multiple_files=False,
    key="jd_match_file",
)
job_role = st.text_input(
    "Job title, if you do not have the posting",
    placeholder="Data analyst, staff nurse, frontend developer",
    key="cv_job_title",
)


def save_upload(upload) -> Path | None:
    if upload is None:
        return None
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.raw_dir / Path(upload.name).name
    destination.write_bytes(upload.getvalue())
    index_file(destination, settings)
    return destination


signature = (
    None if cv_upload is None else (cv_upload.name, cv_upload.size),
    None if jd_upload is None else (jd_upload.name, jd_upload.size),
)
if any(signature) and signature != st.session_state.get("cv_match_signature"):
    try:
        if cv_upload and jd_upload and Path(cv_upload.name).name == Path(jd_upload.name).name:
            raise ValueError("Give the CV and the job description different file names.")
        saved = [path.name for path in (save_upload(cv_upload), save_upload(jd_upload)) if path]
        st.session_state.cv_match_cv = Path(cv_upload.name).name if cv_upload else None
        st.session_state.cv_match_jd = Path(jd_upload.name).name if jd_upload else None
        st.session_state.cv_match_signature = signature
        st.success(f"Indexed {', '.join(saved)}.")
    except Exception as exc:
        st.error(str(exc))

cv_name = st.session_state.get("cv_match_cv")
jd_name = st.session_state.get("cv_match_jd")

if st.button("Match CV to this job description", type="primary", key="cv_match_button"):
    try:
        if not jd_name:
            st.error("Upload a job description first.")
        elif cv_name and not chunks_for_source(settings, cv_name):
            st.error(f"{cv_name} is not in the index yet.")
        else:
            st.session_state.cv_match_report = analyze_sources(settings, cv_name, jd_name)
    except Exception as exc:
        st.error(str(exc))

report = st.session_state.get("cv_match_report")
if report is not None:
    for match in report.matches:
        render_match(match)
    if report.steps:
        st.markdown("**Step-by-step guide**")
        for step in report.steps:
            st.write(step)

if cv_name and st.button("Describe the role from the job title", key="cv_role_button"):
    try:
        if not job_role.strip():
            st.error("Type the job title first.")
        else:
            result = review_cv_for_role(settings, cv_name, job_role)
            st.session_state.cv_role_text = result.text
    except Exception as exc:
        st.error(str(exc))

if st.session_state.get("cv_role_text"):
    st.markdown(st.session_state.cv_role_text)
