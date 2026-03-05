#!/usr/bin/env python3
"""Legacy entrypoint for weak-to-strong addition experiment."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.addition_pipeline import main


if __name__ == "__main__":
    main()
