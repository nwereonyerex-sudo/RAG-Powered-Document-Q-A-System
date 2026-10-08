"""A CV stored for matching."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CV(BaseModel):
    source: str
    text: str = ""
    chunks: int = 0
    skills: list[str] = Field(default_factory=list)
