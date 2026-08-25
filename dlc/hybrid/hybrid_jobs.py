#!/usr/bin/env python3
"""Safe Hybrid worker entry point for reviewed active-learning datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _active_jobs_impl as _impl
from dlc.hybrid._active_jobs_impl import *  # noqa: E402,F401,F403
from dlc.hybrid._hybrid_pipeline_impl import load_box_labels


def pending_active_reviews(cfg: dict) -> int:
    labels = Path(cfg["yolo"]["dataset_dir"]) / "box_labels.csv"
    if not labels.is_file():
        return 0
    rows = load_box_labels(labels)
    if "review_batch" not in rows:
        return 0
    active = rows.review_batch.astype(str).str.startswith("active_")
    return int((active & ~rows.reviewed).sum())


def require_review_complete(cfg: dict) -> None:
    pending = pending_active_reviews(cfg)
    if pending:
        raise RuntimeError(
            f"{pending} active-learning frames are still unsaved. Finish the frame-by-frame review before export or training."
        )


def job_export_reviewed_dataset(cfg: dict) -> None:
    require_review_complete(cfg)
    _impl.ALL_JOBS["export_yolo"](cfg)


def job_fine_tune_yolo(cfg: dict) -> None:
    require_review_complete(cfg)
    dataset = Path(cfg["yolo"]["dataset_dir"])
    labels, yaml = dataset / "box_labels.csv", dataset / "data.yaml"
    if labels.is_file() and (not yaml.is_file() or yaml.stat().st_mtime < labels.stat().st_mtime):
        raise RuntimeError("Labels changed after the last export. Export the combined dataset before fine-tuning.")
    _impl.job_fine_tune_yolo(cfg)


ALL_JOBS = dict(_impl.ALL_JOBS)
ALL_JOBS["export_yolo"] = job_export_reviewed_dataset
ALL_JOBS["fine_tune_yolo"] = job_fine_tune_yolo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(ALL_JOBS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _impl.load_config(args.config)
    print(f"=== {args.action} ===", flush=True)
    ALL_JOBS[args.action](cfg)
    print(f"=== {args.action} completed ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
