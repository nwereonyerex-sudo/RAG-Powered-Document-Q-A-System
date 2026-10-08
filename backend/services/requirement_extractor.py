"""Pull requirement lines out of a job description."""

from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_HEADINGS = ("requirement", "responsibilit", "skill", "qualification", "must")


def _split_items(text: str) -> list[str]:
    cleaned = text.strip().strip(".")
    if not cleaned:
        return []
    if any(mark in cleaned for mark in ",;"):
        parts = re.split(r"[;,]", cleaned)
        return [part.strip(" .") for part in parts if part.strip(" .")]
    return [cleaned]


def _dedupe(items: list[str]) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = " ".join(item.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept[:12]


def extract_requirements_heuristic(text: str) -> list[str]:
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if ":" in line and any(heading in lowered for heading in _HEADINGS):
            items.extend(_split_items(line.split(":", 1)[1]))
            continue
        if line[0] in "-*•":
            items.append(line.lstrip("-*• ").strip())
    if items:
        return _dedupe(items)
    return _dedupe(_split_items(text.replace("\n", ", ")))


def extract_requirements(text: str, llm: BaseChatModel | None = None) -> list[str]:
    heuristic = extract_requirements_heuristic(text)
    if llm is None:
        return heuristic
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "List the distinct requirements written in the job description. "
                "Reply with a JSON array of short strings. Use only the job description.",
            ),
            ("human", "{job_description}"),
        ]
    )
    try:
        raw = (prompt | llm | StrOutputParser()).invoke({"job_description": text[:12000]})
        start = raw.find("[")
        end = raw.rfind("]")
        parsed = json.loads(raw[start : end + 1] if start >= 0 and end > start else raw)
        if isinstance(parsed, list):
            cleaned = [str(item).strip() for item in parsed if str(item).strip()]
            if cleaned:
                return _dedupe(cleaned)
    except Exception:
        return heuristic
    return heuristic
