"""CV matching endpoints. Document questions stay in the Streamlit Q&A page."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.schemas.analysis import Analysis
from backend.services.recommendation_engine import analyze_sources
from rag.config import load_settings

router = APIRouter()


class MatchRequest(BaseModel):
    cv_source: str | None = None
    job_source: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    search_type: str = "mmr"


class Health(BaseModel):
    status: str = Field(default="ok")


@router.get("/health", response_model=Health)
def health() -> Health:
    return Health()


@router.post("/analysis/cv-match", response_model=Analysis)
def cv_match(body: MatchRequest) -> Analysis:
    settings = load_settings(
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        top_k=body.top_k,
        search_type=body.search_type,
    )
    try:
        return analyze_sources(settings, body.cv_source, body.job_source)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
