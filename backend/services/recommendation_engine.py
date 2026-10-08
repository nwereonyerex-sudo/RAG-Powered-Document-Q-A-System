"""Build a requirement-by-requirement CV edit guide."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from backend.schemas.analysis import Analysis, RequirementMatch
from backend.services.cv_matcher import preliminary_match
from backend.services.document_parser import text_for_source
from backend.services.evidence_validator import validate_match
from backend.services.requirement_extractor import extract_requirements
from backend.services.retriever import retrieve_for_requirement
from rag.config import Settings
from rag.chain import build_llm


def _steps(matches: list[RequirementMatch]) -> list[str]:
    steps: list[str] = []
    for index, match in enumerate(matches, start=1):
        if match.status == "met":
            steps.append(
                f'{index}. Leave the line that already shows "{match.requirement}" as it is.'
            )
        elif match.status == "partial":
            steps.append(
                f"{index}. Edit the CV line quoted for {match.requirement}. {match.suggestion}"
            )
        else:
            steps.append(
                f"{index}. Do not add {match.requirement} unless you can describe work you have done."
            )
    return steps


def analyze_texts(
    cv_text: str,
    job_text: str,
    *,
    cv_source: str = "",
    job_source: str = "",
    llm: BaseChatModel | None = None,
) -> Analysis:
    """Match requirements to CV sentences without using the vector index."""
    requirements = extract_requirements(job_text, llm)
    matches = []
    for requirement in requirements:
        status, evidence = preliminary_match(requirement, cv_text)
        matches.append(validate_match(requirement, status, evidence))
    return Analysis(
        cv_source=cv_source,
        job_source=job_source,
        matches=matches,
        steps=_steps(matches),
    )


def analyze_sources(
    settings: Settings,
    cv_source: str | None,
    job_source: str,
    llm: BaseChatModel | None = None,
) -> Analysis:
    """Retrieve CV evidence per requirement, then validate every claim."""
    job_text = text_for_source(settings, job_source)
    if not job_text.strip():
        raise ValueError(f"{job_source} is not in the index, so there are no requirements to match.")
    cv_text = text_for_source(settings, cv_source) if cv_source else ""
    model = llm
    if model is None:
        try:
            model = build_llm(settings)
        except Exception:
            model = None
    requirements = extract_requirements(job_text, model)
    matches = []
    for requirement in requirements:
        retrieved = ""
        if cv_source:
            documents = retrieve_for_requirement(requirement, settings, cv_source)
            retrieved = "\n".join(document.page_content for document in documents)
        status, evidence = preliminary_match(requirement, retrieved or cv_text)
        if status == "missing" and cv_text and retrieved:
            status, evidence = preliminary_match(requirement, cv_text)
        matches.append(validate_match(requirement, status, evidence))
    return Analysis(
        cv_source=cv_source or "",
        job_source=job_source,
        matches=matches,
        steps=_steps(matches),
    )
