"""Retrieve CV chunks for one requirement, separately from a general question."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.config import Settings
from rag.store import retrieve


def retrieve_for_requirement(
    requirement: str,
    settings: Settings,
    cv_source: str,
) -> list[Document]:
    return retrieve(requirement, settings, source_filter=cv_source)
