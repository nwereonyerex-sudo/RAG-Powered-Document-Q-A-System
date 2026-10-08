"""The partial FastAPI match must not be rewritten as FastAPI experience."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.recommendation_engine import analyze_texts

CV = "Developed Python applications and integrated REST APIs."
JOB = "Requirements: FastAPI experience"
SUGGESTION = (
    "If you have actually developed APIs using FastAPI, add the framework and explain what you built. "
    "Otherwise, retain the verified REST API experience without claiming FastAPI proficiency."
)


class FastApiPartialMatchTests(unittest.TestCase):
    def test_partial_match_keeps_rest_api_evidence(self) -> None:
        report = analyze_texts(CV, JOB)
        self.assertEqual(len(report.matches), 1)
        match = report.matches[0]
        self.assertEqual(match.requirement, "FastAPI experience")
        self.assertEqual(match.status, "partial")
        self.assertEqual(match.cv_evidence, CV)
        self.assertEqual(match.suggestion, SUGGESTION)
        self.assertNotIn("you have FastAPI", match.suggestion)

    def test_quoted_evidence_is_in_the_cv(self) -> None:
        report = analyze_texts(CV, JOB)
        for match in report.matches:
            if match.cv_evidence:
                self.assertIn(match.cv_evidence, CV)


if __name__ == "__main__":
    unittest.main()
