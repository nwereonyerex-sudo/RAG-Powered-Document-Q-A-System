"""Check that known facts land in the top retrieved chunks."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import load_settings
from rag.store import index_directory, retrieve

CASES = [
    (
        "What was Acme Robotics Q3 revenue?",
        "42 million",
        "acme_q3_report.txt",
    ),
    (
        "Which supplier provides actuator components?",
        "Northwind Steel",
        "sample_filing.html",
    ),
    (
        "How much does SparseRoute cut index size?",
        "35 percent",
        "sample_paper.md",
    ),
]


def main() -> int:
    samples = ROOT / "samples"
    with tempfile.TemporaryDirectory() as tmp:
        settings = load_settings(chroma_dir=Path(tmp), search_type="similarity", top_k=4)
        indexed = index_directory(samples, settings)
        print(f"Indexed {indexed} chunks from {samples}")
        failures = 0
        for question, expected, source in CASES:
            docs = retrieve(question, settings)
            blob = "\n".join(doc.page_content for doc in docs)
            found_text = expected.lower() in blob.lower()
            found_source = any(doc.metadata.get("source") == source for doc in docs)
            status = "PASS" if found_text and found_source else "FAIL"
            if status == "FAIL":
                failures += 1
            print(f"{status}: {question}")
            print(f"  expected snippet: {expected} from {source}")
            for doc in docs:
                preview = " ".join(doc.page_content.split())[:120]
                print(f"  - {doc.metadata.get('source')}: {preview}")
        if failures:
            print(f"{failures} retrieval check(s) failed.")
            return 1
        print("All retrieval checks passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
