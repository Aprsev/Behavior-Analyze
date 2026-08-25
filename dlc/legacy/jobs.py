#!/usr/bin/env python3
"""Package-aware entry point for the original GUI background jobs."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.legacy import _jobs_impl as _impl

_impl.ROOT = REPOSITORY_ROOT

from dlc.legacy._jobs_impl import *  # noqa: E402,F401,F403


if __name__ == "__main__":
    raise SystemExit(_impl.main())
