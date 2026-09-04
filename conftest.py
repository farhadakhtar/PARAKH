"""Pytest bootstrap: put the project root on ``sys.path``.

Keeps ``import src.stage1...`` working regardless of the directory pytest is
invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
