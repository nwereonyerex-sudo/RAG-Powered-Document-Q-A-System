"""Compare one job requirement with the text retrieved from a CV."""

from __future__ import annotations

import re

from backend.services.evidence_validator import skill_terms

_RELATED = {
    "fastapi": ("rest", "api", "apis", "http", "endpoint"),
    "django": ("python", "web"),
    "flask": ("python", "api", "apis"),
    "react": ("javascript", "frontend", "ui"),
    "sql": ("database", "query", "postgres", "mysql"),
    "aws": ("cloud", "s3", "ec2"),
}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _score(requirement: str, sentence: str) -> int:
    terms = skill_terms(requirement)
    lowered = sentence.lower()
    score = sum(2 for term in terms if term.lower() in lowered)
    for term in terms:
        for hint in _RELATED.get(term.lower(), ()):
            if hint in lowered:
                score += 1
    return score


def preliminary_match(requirement: str, cv_text: str) -> tuple[str, str]:
    """Return a status and the CV sentence that best supports it."""
    choices = sentences(cv_text)
    if not choices:
        return "missing", ""
    ranked = sorted(choices, key=lambda sentence: _score(requirement, sentence), reverse=True)
    best = ranked[0]
    if _score(requirement, best) <= 0:
        return "missing", ""
    terms = skill_terms(requirement)
    if terms and all(term.lower() in best.lower() for term in terms):
        return "met", best
    return "partial", best
