"""Checks that CV quotes used in a match actually appear in the CV."""

from __future__ import annotations

from backend.schemas.analysis import Analysis


def supported_quote_rate(analysis: Analysis, cv_text: str) -> float:
    quoted = [match for match in analysis.matches if match.cv_evidence]
    if not quoted:
        return 1.0
    supported = sum(1 for match in quoted if match.cv_evidence in cv_text)
    return supported / len(quoted)


def unsupported_claims(analysis: Analysis, forbidden: str) -> list[str]:
    needle = forbidden.lower()
    return [
        match.requirement
        for match in analysis.matches
        if needle in match.suggestion.lower()
    ]
