"""Load CV and job-description files with the same parsers as document Q&A."""

from __future__ import annotations

from pathlib import Path

from rag.config import Settings
from rag.ingest import SUPPORTED, load_file
from rag.store import chunks_for_source, index_paths


def supported_suffixes() -> set[str]:
    return set(SUPPORTED)


def parse_file(path: Path):
    return load_file(path)


def index_file(path: Path, settings: Settings) -> int:
    return index_paths([path], settings)


def text_for_source(settings: Settings, source_name: str) -> str:
    documents = chunks_for_source(settings, source_name)
    return "\n".join(document.page_content for document in documents if document.page_content.strip())
