#!/usr/bin/env python3
"""Hybrid worker entry point with portable Ultralytics device selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _review_guard_jobs as _base
from dlc.hybrid._review_guard_jobs import *  # noqa: E402,F401,F403
from dlc.hybrid.device import normalize_yolo_device


def _run(action: str, cfg: dict) -> None:
    # Resolve the GUI's portable "auto" value only in this worker process.  The
    # saved JSON remains portable between CUDA, Apple Silicon, and CPU hosts.
    normalize_yolo_device(cfg)
    _base.ALL_JOBS[action](cfg)


def _wrapped(action: str) -> Callable[[dict], None]:
    def run(cfg: dict) -> None:
        _run(action, cfg)

    run.__name__ = getattr(_base.ALL_JOBS[action], "__name__", f"job_{action}")
    return run


ALL_JOBS = {name: _wrapped(name) for name in _base.ALL_JOBS}


# Keep direct imports used by integrations and tests on the same safe path.
job_train_yolo = ALL_JOBS["train_yolo"]
job_validate_yolo = ALL_JOBS["validate_yolo"]
job_prepare_hybrid = ALL_JOBS["prepare_hybrid"]
job_full_hybrid = ALL_JOBS["full_hybrid"]
job_mine_active_frames = ALL_JOBS["mine_active_frames"]
job_fine_tune_yolo = ALL_JOBS["fine_tune_yolo"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(ALL_JOBS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _base._impl.load_config(args.config)
    print(f"=== {args.action} ===", flush=True)
    ALL_JOBS[args.action](cfg)
    print(f"=== {args.action} completed ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
