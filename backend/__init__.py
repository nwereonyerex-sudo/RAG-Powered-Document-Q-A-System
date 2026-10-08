"""API and CV-matching services. Document Q&A stays in src/rag."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)
