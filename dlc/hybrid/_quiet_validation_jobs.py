#!/usr/bin/env python3
"""Hybrid worker with compact training and quiet validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _active_jobs_impl as _config
from dlc.hybrid import _compact_train_jobs as _base
from dlc.hybrid import _hybrid_jobs_impl as _initial
from dlc.hybrid._compact_train_jobs import *  # noqa: E402,F401,F403
from dlc.hybrid.device import normalize_yolo_device


def job_validate_yolo(cfg: dict) -> None:
    from ultralytics import YOLO

    normalize_yolo_device(cfg)
    yolo = cfg["yolo"]
    checkpoint = Path(yolo["trained_model"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    kwargs = {
        "data": str(Path(yolo["dataset_dir"]) / "data.yaml"),
        "imgsz": int(yolo.get("image_size", 640)),
        "batch": int(yolo.get("batch_size", 8)),
        "device": yolo.get("device", 0), "plots": True,
    }
    kwargs.update(_initial._yolo_advanced(cfg, "validate_yolo"))
    kwargs["verbose"] = False
    metrics = YOLO(str(checkpoint)).val(**kwargs)
    print(
        "YOLO validation summary: "
        f"mAP50-95={float(metrics.box.map):.6f}; "
        f"mAP50={float(metrics.box.map50):.6f}; "
        f"precision={float(metrics.box.mp):.6f}; "
        f"recall={float(metrics.box.mr):.6f}",
        flush=True,
    )


ALL_JOBS = dict(_base.ALL_JOBS)
ALL_JOBS["validate_yolo"] = job_validate_yolo


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
