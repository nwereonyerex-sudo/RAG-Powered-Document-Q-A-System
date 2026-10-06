"""Load PDF, plain text, and HTML documents and split them into chunks."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings

SUPPORTED = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
}


def iter_supported(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    doc_type = SUPPORTED.get(suffix)
    if doc_type is None:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use PDF, TXT, MD, or HTML."
        )

    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    elif suffix in {".html", ".htm"}:
        documents = BSHTMLLoader(
            str(path),
            open_encoding="utf-8",
            bs_kwargs={"features": "html.parser"},
        ).load()
    else:
        documents = TextLoader(str(path), encoding="utf-8").load()

    for document in documents:
        document.metadata["source"] = path.name
        document.metadata["doc_type"] = doc_type
        page = document.metadata.get("page", 0)
        document.metadata["page"] = int(page) if page is not None else 0
    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page", 0)
        chunk.metadata = {
            "source": source,
            "doc_type": str(chunk.metadata.get("doc_type", "text")),
            "page": int(page) if isinstance(page, int) else 0,
            "chunk_id": f"{source}:{page}:{index}",
        }
    return chunks
