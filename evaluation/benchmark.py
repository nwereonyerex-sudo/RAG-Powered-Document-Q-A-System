"""Score the specialised matcher on saved cases. Direct LLM comparison stays optional."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.recommendation_engine import analyze_texts
from evaluation.retrieval_metrics import supported_quote_rate, unsupported_claims

DATASETS = Path(__file__).resolve().parent / "datasets"


def score_case(path: Path) -> tuple[bool, str]:
    case = json.loads(path.read_text(encoding="utf-8"))
    report = analyze_texts(case["cv"], case["job_description"])
    if not report.matches:
        return False, f"{path.name}: no requirements extracted"
    match = report.matches[0]
    quote_rate = supported_quote_rate(report, case["cv"])
    problems = []
    if match.status != case["expected_status"]:
        problems.append(f"status {match.status} != {case['expected_status']}")
    if quote_rate < 1:
        problems.append("a quote is not in the CV")
    forbidden = unsupported_claims(report, case.get("forbidden_claim", ""))
    if case.get("forbidden_claim") and forbidden:
        problems.append(f"forbidden claim on {', '.join(forbidden)}")
    if problems:
        return False, f"{path.name}: {'; '.join(problems)}"
    return True, f"{path.name}: {match.status}"


def main() -> int:
    cases = sorted(DATASETS.glob("*.json"))
    if not cases:
        print("No cases in evaluation/datasets.")
        return 1
    failures = 0
    for path in cases:
        ok, message = score_case(path)
        print(("PASS " if ok else "FAIL ") + message)
        failures += not ok
    print(
        "Direct LLM comparison is separate: pass an llm into "
        "rag.chain.guide_cv_from_job_description and compare it with analyze_texts."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
