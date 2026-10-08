"""Grounded question answering over retrieved chunks."""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.config import Settings
from rag.store import chunks_for_source, retrieve

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
    )
    return _visible_answer(text) or REFUSAL


def _visible_answer(text: str) -> str:
    """Keep the assistant reply when a local chat model echoes the prompt."""
    marker = "<|im_start|>assistant"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


REVIEW_PROMPT = """You are reviewing one uploaded document. Read the whole document below.

Write exactly two sections with these headings:

Content
Explain what the document is and what it contains. Cover the whole document in plain English. Use only facts written in the document.

How to make it better
Give specific changes that would make this document clearer or stronger. For each suggestion, name the part of the document it applies to. Do not invent jobs, dates, numbers, or credentials that are not in the document.

Document:
{context}
"""


def describe_document(
    settings: Settings,
    source_name: str,
    llm: BaseChatModel | None = None,
) -> Answer:
    """Summarize one whole file and suggest how to improve it."""
    context, documents = _document_context(settings, source_name)
    if not documents:
        return Answer(
            text=f"{source_name} is not in the index, so there is no content to review.",
            sources=[],
        )
    model = llm or build_llm(settings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REVIEW_PROMPT),
            ("human", "Review this document."),
        ]
    )
    text = _visible_answer(
        (prompt | model | StrOutputParser()).invoke({"context": context})
    )
    return Answer(text=text or REFUSAL, sources=documents[:1])


ROLE_PROMPT = """You are a CV editor. The candidate's CV is below. The target job is: {role}

Write exactly three sections with these headings:

What this CV shows
Summarize the experience, skills, and education written in the CV. Do not add employers, dates, or credentials that are not in the CV.

How to improve this CV for this role
Give specific edits: what to add, cut, reorder, or rephrase so the CV speaks to this job. Name the part of the CV each suggestion refers to. If a skill this role usually needs is missing, say it is missing. Do not invent jobs or achievements.

What to expect in this role
Describe the typical day-to-day work, skills, and interview topics for this job. This section is general knowledge about the role, not facts from the CV. Keep it concrete.

CV:
{context}
"""


def _document_context(settings: Settings, source_name: str) -> tuple[str, list[Document]]:
    documents = chunks_for_source(settings, source_name)
    context = format_context(documents)
    if len(context) > 24000:
        context = context[:24000] + "\n\n[The rest of the file was cut for length.]"
    return context, documents


def review_cv_for_role(
    settings: Settings,
    source_name: str,
    role: str,
    llm: BaseChatModel | None = None,
) -> Answer:
    """Review a CV against one job and describe what that job involves."""
    role = " ".join(role.split())
    if not role:
        return Answer(text="Type a job role first.", sources=[])
    context, documents = _document_context(settings, source_name)
    if not documents:
        return Answer(
            text=f"{source_name} is not in the index, so the CV cannot be reviewed.",
            sources=[],
        )
    model = llm or build_llm(settings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROLE_PROMPT),
            ("human", "Review this CV for the role."),
        ]
    )
    text = _visible_answer(
        (prompt | model | StrOutputParser()).invoke({"context": context, "role": role})
    )
    return Answer(text=text or REFUSAL, sources=documents[:1])


def wants_document_review(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    cues = (
        "content",
        "what is this",
        "what is the file",
        "what's in",
        "whats in",
        "summar",
        "about this",
        "about the file",
        "about the document",
        "make it better",
        "make this better",
        "improve",
        "how to make",
    )
    return any(cue in normalized for cue in cues)


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
