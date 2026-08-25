#!/usr/bin/env python3
"""Train/test-aware background jobs for the Hybrid Workbench."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _hybrid_jobs_impl as original
from dlc.hybrid.dataset_config import (
    config_for_record, evaluation_records, load_hybrid_config, records, validate_files,
)
from dlc.hybrid.multi_video_pipeline import export_grouped_yolo_dataset, generate_training_boxes


original.load_config = load_hybrid_config
original.generate_traditional_boxes = generate_training_boxes
original.export_yolo_dataset = export_grouped_yolo_dataset


def job_hybrid_check(cfg: dict) -> None:
    original.job_hybrid_check(cfg)
    print(f"Training videos: {len(records(cfg, 'train'))}")
    print(f"Held-out test videos: {len(records(cfg, 'test'))}")
    problems = validate_files(cfg)
    if problems:
        raise FileNotFoundError("Dataset configuration problems:\n- " + "\n- ".join(problems))
    for split in ("train", "test"):
        for index, item in enumerate(records(cfg, split), 1):
            print(f"{split}[{index}] video OK: {item['video']}")
            print(f"{split}[{index}] ROI OK: {item['roi_json']}")


def _run_records(cfg: dict, selected: list[dict[str, str]], action: Callable[[dict], None]) -> None:
    if not selected:
        raise ValueError("The selected video set is empty")
    for index, item in enumerate(selected, 1):
        print(f"--- {item['split']} video {index}/{len(selected)}: {item['video']} ---", flush=True)
        action(config_for_record(cfg, item))


def _evaluation(cfg: dict, action: Callable[[dict], None]) -> None:
    selected = evaluation_records(cfg)
    if not records(cfg, "test"):
        print("No held-out test videos configured; using training videos for pipeline QA.", flush=True)
    _run_records(cfg, selected, action)


def job_prepare_evaluation(cfg: dict) -> None:
    _evaluation(cfg, original.job_prepare_hybrid)


def job_run_evaluation_dlc(cfg: dict) -> None:
    _evaluation(cfg, original.job_run_hybrid_dlc)


def job_postprocess_evaluation(cfg: dict) -> None:
    _evaluation(cfg, original.job_postprocess_hybrid)


def job_full_evaluation(cfg: dict) -> None:
    _evaluation(cfg, original.job_full_hybrid)


def job_full_train(cfg: dict) -> None:
    _run_records(cfg, records(cfg, "train"), original.job_full_hybrid)


def job_full_test(cfg: dict) -> None:
    selected = records(cfg, "test")
    if not selected:
        raise ValueError("No held-out test videos configured")
    _run_records(cfg, selected, original.job_full_hybrid)


ALL_JOBS = dict(original.ALL_JOBS)
ALL_JOBS.update({
    "hybrid_check": job_hybrid_check,
    "prepare_hybrid": job_prepare_evaluation,
    "run_hybrid_dlc": job_run_evaluation_dlc,
    "postprocess_hybrid": job_postprocess_evaluation,
    "full_hybrid": job_full_evaluation,
    "full_hybrid_train": job_full_train,
    "full_hybrid_test": job_full_test,
})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(ALL_JOBS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_hybrid_config(args.config)
    print(f"=== {args.action} ===", flush=True)
    ALL_JOBS[args.action](cfg)
    print(f"=== {args.action} completed ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
