"""Grounded question answering over retrieved chunks."""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.config import Settings
from rag.store import retrieve

SYSTEM_PROMPT = """You are a document analyst. Answer the question using only the context.

Rules:
- If the context does not contain the answer, reply exactly: "I cannot find that in the uploaded documents."
- Do not add facts from outside the context.
- Keep the answer concise.
- End with a Sources section. Each source line names the file and the page number from the context labels.

Context:
{context}
"""

REFUSAL = "I cannot find that in the uploaded documents."


@dataclass
class Answer:
    text: str
    sources: list[Document]


def format_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for document in documents:
        page = document.metadata.get("page", 0)
        page_number = int(page) + 1 if isinstance(page, int) else page
        blocks.append(
            "\n".join(
                [
                    f"Source file: {document.metadata.get('source', 'unknown')}",
                    f"Page: {page_number}",
                    f"Document type: {document.metadata.get('doc_type', 'text')}",
                    document.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_llm(settings: Settings) -> BaseChatModel:
    provider = settings.resolve_provider()
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or set LLM_PROVIDER=local."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=0)

    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    chat_pipeline = HuggingFacePipeline.from_model_id(
        model_id=settings.local_model,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 400, "do_sample": False},
    )
    return ChatHuggingFace(llm=chat_pipeline)


def generate_answer(
    question: str,
    documents: list[Document],
    llm: BaseChatModel,
) -> str:
    if not documents:
        return REFUSAL
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    text = chain.invoke(
        {"context": format_context(documents), "question": question}
    ).strip()
    return text or REFUSAL


def ask(
    question: str,
    settings: Settings,
    source_filter: str | None = None,
    llm: BaseChatModel | None = None,
) -> Answer:
    documents = retrieve(question, settings, source_filter=source_filter)
    model = llm or build_llm(settings)
    return Answer(
        text=generate_answer(question, documents, model),
        sources=documents,
    )
