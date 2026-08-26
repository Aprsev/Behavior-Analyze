#!/usr/bin/env python3
"""Public hybrid worker with synchronized supervised DLC training videos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _active_jobs_impl as _config
from dlc.hybrid import _napari_safe_jobs as _base
from dlc.hybrid._napari_safe_jobs import *  # noqa: E402,F401,F403
from dlc.hybrid.project_config import resolve_project_config
from dlc.hybrid.project_videos import sync_training_videos


def job_extract_frames(cfg: dict) -> None:
    project = resolve_project_config(cfg)
    sync_training_videos(cfg, project)
    _base.ALL_JOBS["extract_frames"](cfg)


ALL_JOBS = dict(_base.ALL_JOBS)
ALL_JOBS["extract_frames"] = job_extract_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(ALL_JOBS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _config.load_config(args.config)
    print(f"=== {args.action} ===", flush=True)
    ALL_JOBS[args.action](cfg)
    print(f"=== {args.action} completed ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
