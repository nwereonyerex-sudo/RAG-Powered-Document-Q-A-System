"""Structured CV-to-job analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MatchStatus = Literal["met", "partial", "missing"]


class RequirementMatch(BaseModel):
    requirement: str
    status: MatchStatus
    cv_evidence: str = ""
    suggestion: str


class Analysis(BaseModel):
    cv_source: str = ""
    job_source: str = ""
    matches: list[RequirementMatch] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
