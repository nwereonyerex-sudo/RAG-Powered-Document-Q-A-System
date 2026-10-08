"""Persistent Chroma index and retrieval."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import Settings
from rag.ingest import iter_supported, load_file, split_documents

_EMBEDDINGS: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    cached = _EMBEDDINGS.get(settings.embedding_model)
    if cached is None:
        cached = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _EMBEDDINGS[settings.embedding_model] = cached
    return cached


def open_store(settings: Settings) -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
    )


def _drop_source(store: Chroma, source_name: str) -> None:
    existing = store.get(where={"source": source_name})
    ids = existing.get("ids") or []
    if ids:
        store.delete(ids=ids)


def index_paths(paths: list[Path], settings: Settings) -> int:
    """Replace any previous chunks for each file, then store the new chunks."""
    store = open_store(settings)
    total = 0
    for path in paths:
        documents = load_file(path)
        chunks = split_documents(documents, settings)
        _drop_source(store, path.name)
        if chunks:
            store.add_documents(chunks)
            total += len(chunks)
    return total


def index_directory(directory: Path, settings: Settings) -> int:
    return index_paths(iter_supported(directory), settings)


def list_sources(settings: Settings) -> list[str]:
    store = open_store(settings)
    found = store.get()
    names: list[str] = []
    seen: set[str] = set()
    for meta in found.get("metadatas") or []:
        name = str((meta or {}).get("source", "")).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def chunks_for_source(settings: Settings, source_name: str) -> list[Document]:
    """Return every stored chunk for one file, in page order."""
    store = open_store(settings)
    found = store.get(where={"source": source_name})
    rows = zip(
        found.get("ids") or [],
        found.get("metadatas") or [],
        found.get("documents") or [],
    )
    documents = [
        Document(page_content=text or "", metadata=meta or {})
        for _, meta, text in rows
        if (text or "").strip()
    ]
    documents.sort(
        key=lambda doc: (
            int(doc.metadata.get("page", 0) or 0),
            str(doc.metadata.get("chunk_id", "")),
        )
    )
    return documents


def retrieve(
    question: str,
    settings: Settings,
    source_filter: str | None = None,
) -> list[Document]:
    store = open_store(settings)
    metadata_filter = {"source": source_filter} if source_filter else None
    if settings.search_type == "mmr":
        return store.max_marginal_relevance_search(
            question,
            k=settings.top_k,
            fetch_k=max(20, settings.top_k * 5),
            filter=metadata_filter,
        )

    if settings.score_threshold > 0:
        ranked = store.similarity_search_with_relevance_scores(
            question,
            k=settings.top_k,
            filter=metadata_filter,
        )
        return [doc for doc, score in ranked if score >= settings.score_threshold]

    return store.similarity_search(
        question,
        k=settings.top_k,
        filter=metadata_filter,
    )
