#!/usr/bin/env python3
"""Compatibility launcher for the modular environment doctor."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.tools.environment_doctor import main


if __name__ == "__main__":
    raise SystemExit(main())
