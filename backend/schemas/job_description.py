"""A job description and the requirements taken from it."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    source: str
    text: str = ""
    requirements: list[str] = Field(default_factory=list)
