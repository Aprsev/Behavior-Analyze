#!/usr/bin/env python3
"""Public hybrid worker with supervised DLC project auto-discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _active_jobs_impl as _config
from dlc.hybrid import _video_safe_jobs as _base
from dlc.hybrid._video_safe_jobs import *  # noqa: E402,F401,F403
from dlc.hybrid.project_config import resolve_project_config

PROJECT_REQUIRED_ACTIONS = {
    "add_videos", "extract_frames", "label_frames", "check_labels",
    "build_dataset", "train", "evaluate", "analyze", "filter_predictions",
    "create_labeled_video", "plot_trajectories", "extract_outliers",
    "refine_labels", "merge_datasets",
}


def _project_action(name: str) -> Callable[[dict], None]:
    def run(cfg: dict) -> None:
        path = resolve_project_config(cfg)
        print(
            "HYBRID_GUI_RESULT " + json.dumps({"project_config": str(path)}, ensure_ascii=False),
            flush=True,
        )
        print(f"Using DLC project: {path}", flush=True)
        _base.ALL_JOBS[name](cfg)

    run.__name__ = getattr(_base.ALL_JOBS[name], "__name__", f"job_{name}")
    return run


ALL_JOBS = dict(_base.ALL_JOBS)
for _name in PROJECT_REQUIRED_ACTIONS.intersection(ALL_JOBS):
    ALL_JOBS[_name] = _project_action(_name)


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
