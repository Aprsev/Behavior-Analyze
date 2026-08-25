#!/usr/bin/env python3
"""Hybrid worker entry point with compact YOLO training logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _active_jobs_impl as _active
from dlc.hybrid import _device_jobs as _base
from dlc.hybrid import _hybrid_jobs_impl as _initial
from dlc.hybrid._device_jobs import *  # noqa: E402,F401,F403
from dlc.hybrid.compact_yolo import compact_train
from dlc.hybrid.device import normalize_yolo_device


def job_train_yolo(cfg: dict) -> None:
    from ultralytics import YOLO

    normalize_yolo_device(cfg)
    yolo = cfg["yolo"]
    data = Path(yolo["dataset_dir"]) / "data.yaml"
    if not data.is_file():
        raise FileNotFoundError("Export the reviewed YOLO dataset first")
    kwargs = {
        "data": str(data), "epochs": int(yolo.get("epochs", 100)),
        "imgsz": int(yolo.get("image_size", 640)), "batch": int(yolo.get("batch_size", 8)),
        "device": yolo.get("device", 0), "workers": int(yolo.get("workers", 4)),
        "patience": int(yolo.get("patience", 30)),
        "project": str(Path(yolo["dataset_dir"]) / "runs"),
        "name": yolo.get("run_name", "mouse_detector"), "exist_ok": True,
        "seed": int(yolo.get("split_seed", 42)), "plots": True,
    }
    kwargs.update(_initial._yolo_advanced(cfg, "train_yolo"))
    result = compact_train(YOLO(yolo.get("base_model", "yolo26n.pt")), kwargs, 5)
    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Best YOLO checkpoint: {best}", flush=True)
    _initial.emit(yolo_trained_model=str(best.resolve()))


def job_fine_tune_yolo(cfg: dict) -> None:
    from ultralytics import YOLO

    normalize_yolo_device(cfg)
    _base.require_review_complete(cfg)
    yolo, active = cfg.get("yolo", {}), cfg.get("active_learning", {})
    checkpoint = Path(yolo.get("trained_model", "")).expanduser().resolve()
    dataset = Path(yolo["dataset_dir"]).expanduser().resolve()
    data, labels = dataset / "data.yaml", dataset / "box_labels.csv"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Existing YOLO checkpoint is missing: {checkpoint}")
    if not data.is_file():
        raise FileNotFoundError("Export the combined reviewed dataset before fine-tuning")
    if labels.is_file() and data.stat().st_mtime < labels.stat().st_mtime:
        raise RuntimeError("Labels changed after the last export. Export the combined dataset before fine-tuning.")
    kwargs = {
        "data": str(data), "epochs": int(active.get("fine_tune_epochs", 50)),
        "imgsz": int(yolo.get("image_size", 640)), "batch": int(yolo.get("batch_size", 8)),
        "device": yolo.get("device", 0), "workers": int(yolo.get("workers", 4)),
        "patience": int(yolo.get("patience", 30)), "lr0": float(active.get("learning_rate", .001)),
        "project": str(dataset / "runs"), "name": active.get("run_name", "mouse_detector_active"),
        "exist_ok": True, "seed": int(yolo.get("split_seed", 42)), "plots": True, "resume": False,
    }
    kwargs.update(_initial._yolo_advanced(cfg, "fine_tune_yolo"))
    print(f"Initializing active-learning fine-tune from: {checkpoint}", flush=True)
    result = compact_train(YOLO(str(checkpoint)), kwargs, 5)
    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Fine-tuned YOLO checkpoint: {best}", flush=True)
    _active.emit(yolo_trained_model=str(best.resolve()))


ALL_JOBS = dict(_base.ALL_JOBS)
ALL_JOBS["train_yolo"] = job_train_yolo
ALL_JOBS["fine_tune_yolo"] = job_fine_tune_yolo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(ALL_JOBS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _base._base._impl.load_config(args.config)
    print(f"=== {args.action} ===", flush=True)
    ALL_JOBS[args.action](cfg)
    print(f"=== {args.action} completed ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
