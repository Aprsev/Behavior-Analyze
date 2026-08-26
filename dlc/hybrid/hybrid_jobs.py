#!/usr/bin/env python3
"""Public hybrid worker with verified extraction and compatible napari labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _active_jobs_impl as _config
from dlc.hybrid import _project_safe_jobs as _base
from dlc.hybrid._project_safe_jobs import *  # noqa: E402,F401,F403
from dlc.hybrid.labeling_data import choose_label_folder, extraction_summary
from dlc.hybrid.project_config import resolve_project_config


def _emit_project(path: Path) -> None:
    print(
        "HYBRID_GUI_RESULT " + json.dumps({"project_config": str(path)}, ensure_ascii=False),
        flush=True,
    )
    print(f"Using DLC project: {path}", flush=True)


def job_extract_frames(cfg: dict) -> None:
    _base.ALL_JOBS["extract_frames"](cfg)
    project = resolve_project_config(cfg)
    print(extraction_summary(project), flush=True)


def job_label_frames(cfg: dict) -> None:
    project = resolve_project_config(cfg)
    _emit_project(project)
    selected, valid = choose_label_folder(project)
    remaining = sum(not folder.has_collected_data for folder in valid)
    print(
        f"Opening napari folder: {selected.path} ({selected.image_count} images; "
        f"{remaining} unlabeled folders remain)",
        flush=True,
    )
    # Some DLC releases pass only the image folder, while current
    # napari-deeplabcut expects the project config alongside a new image folder.
    from deeplabcut.gui.widgets import launch_napari

    launch_napari(files=[str(selected.path), str(project)])


ALL_JOBS = dict(_base.ALL_JOBS)
ALL_JOBS["extract_frames"] = job_extract_frames
ALL_JOBS["label_frames"] = job_label_frames


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
