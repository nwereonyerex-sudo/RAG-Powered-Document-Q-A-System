"""Stop a match from claiming a skill the CV does not show."""

from __future__ import annotations

import re

from backend.schemas.analysis import MatchStatus, RequirementMatch

_STOP = {
    "a",
    "an",
    "and",
    "experience",
    "for",
    "in",
    "knowledge",
    "of",
    "or",
    "proficient",
    "proficiency",
    "skill",
    "skills",
    "the",
    "using",
    "with",
    "years",
    "year",
}


def skill_terms(requirement: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+#./-]*", requirement):
        if token.lower() in _STOP or len(token) < 2:
            continue
        if token.lower() not in {item.lower() for item in terms}:
            terms.append(token)
    return terms


def _contains(term: str, text: str) -> bool:
    return term.lower() in text.lower()


def _partial_suggestion(missing: list[str], evidence: str) -> str:
    skill = missing[0]
    if skill.lower() == "fastapi" and "rest" in evidence.lower():
        return (
            "If you have actually developed APIs using FastAPI, add the framework and explain what you built. "
            "Otherwise, retain the verified REST API experience without claiming FastAPI proficiency."
        )
    names = " and ".join(missing)
    verified = evidence.strip()
    return (
        f"If you have actually used {names}, add it and explain what you built. "
        f'Otherwise, retain the verified experience ("{verified}") without claiming {names} proficiency.'
    )


def validate_match(
    requirement: str,
    status: str,
    cv_evidence: str,
    suggestion: str = "",
) -> RequirementMatch:
    """Keep the status and the wording inside what the CV quote supports."""
    evidence = cv_evidence.strip().strip('"').strip()
    terms = skill_terms(requirement)
    missing = [term for term in terms if not _contains(term, evidence)]
    present = [term for term in terms if _contains(term, evidence)]
    normalized = status if status in {"met", "partial", "missing"} else "partial"

    if not evidence:
        final: MatchStatus = "missing"
        text = (
            f"The CV does not show {requirement}. "
            "Add it only if you have actually done the work, and describe what you built. "
            "Do not claim it otherwise."
        )
    elif missing:
        final = "partial"
        text = _partial_suggestion(missing, evidence)
    elif present or normalized == "met":
        final = "met"
        text = suggestion.strip() or (
            f'This requirement is already shown. Keep the wording close to the CV: "{evidence}".'
        )
    else:
        final = "partial" if evidence else "missing"
        text = suggestion.strip() or _partial_suggestion(terms or [requirement], evidence)

    return RequirementMatch(
        requirement=requirement.strip(),
        status=final,
        cv_evidence=evidence,
        suggestion=text,
    )
