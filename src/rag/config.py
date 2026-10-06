"""Runtime settings for chunking, retrieval, and the LLM provider."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class Settings:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    search_type: str = "mmr"
    score_threshold: float = 0.0
    llm_provider: str = "auto"
    openai_model: str = "gpt-4o-mini"
    local_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: Path = ROOT / "data" / "chroma"
    raw_dir: Path = ROOT / "data" / "raw"

    def __post_init__(self) -> None:
        if self.chunk_size < 200:
            raise ValueError("chunk_size must be at least 200 characters.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.search_type not in {"mmr", "similarity"}:
            raise ValueError("search_type must be 'mmr' or 'similarity'.")
        if self.llm_provider not in {"auto", "openai", "local"}:
            raise ValueError("llm_provider must be 'auto', 'openai', or 'local'.")
        self.chroma_dir = Path(self.chroma_dir)
        self.raw_dir = Path(self.raw_dir)

    @property
    def collection_name(self) -> str:
        """Keep each splitter configuration in its own collection."""
        return f"docs_minilm_c{self.chunk_size}_o{self.chunk_overlap}"

    def resolve_provider(self) -> str:
        if self.llm_provider == "openai":
            return "openai"
        if self.llm_provider == "local":
            return "local"
        if os.getenv("OPENAI_API_KEY", "").strip():
            return "openai"
        return "local"


def load_settings(**overrides: object) -> Settings:
    settings = Settings(
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        top_k=_env_int("TOP_K", 4),
        search_type=os.getenv("SEARCH_TYPE", "mmr"),
        score_threshold=_env_float("SCORE_THRESHOLD", 0.0),
        llm_provider=os.getenv("LLM_PROVIDER", "auto"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        local_model=os.getenv("LOCAL_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    settings.__post_init__()
    return settings
