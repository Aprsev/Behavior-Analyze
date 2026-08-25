#!/usr/bin/env python3
"""Hybrid jobs extended with YOLO active learning and checkpoint fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _train_test_jobs as base
from dlc.hybrid.active_learning import import_labeled_dataset, mine_low_confidence_frames


def load_config(path: Path) -> dict:
    cfg = base.load_hybrid_config(path)
    active = cfg.setdefault("active_learning", {})
    config_dir = path.resolve().parent
    if active.get("import_dataset_dir"):
        value = Path(active["import_dataset_dir"]).expanduser()
        active["import_dataset_dir"] = str(value if value.is_absolute() else (config_dir / value).resolve())
    active["new_videos"] = [
        str(value if value.is_absolute() else (config_dir / value).resolve())
        for raw in active.get("new_videos", [])
        for value in [Path(raw).expanduser()]
    ]
    return cfg


def emit(**values) -> None:
    base.original.emit(**values)


def job_import_labeled_dataset(cfg: dict) -> None:
    report = import_labeled_dataset(cfg)
    emit(active_import=report)


def job_mine_active_frames(cfg: dict) -> None:
    report = mine_low_confidence_frames(cfg)
    emit(active_batch=report)


def job_fine_tune_yolo(cfg: dict) -> None:
    from ultralytics import YOLO

    yolo, active = cfg.get("yolo", {}), cfg.get("active_learning", {})
    checkpoint = Path(yolo.get("trained_model", "")).expanduser().resolve()
    data = Path(yolo["dataset_dir"]).expanduser().resolve() / "data.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Existing YOLO checkpoint is missing: {checkpoint}")
    if not data.is_file():
        raise FileNotFoundError("Export the combined reviewed dataset before fine-tuning")
    kwargs = {
        "data": str(data),
        "epochs": int(active.get("fine_tune_epochs", 50)),
        "imgsz": int(yolo.get("image_size", 640)),
        "batch": int(yolo.get("batch_size", 8)),
        "device": yolo.get("device", "auto"),
        "workers": int(yolo.get("workers", 4)),
        "patience": int(yolo.get("patience", 30)),
        "lr0": float(active.get("learning_rate", .001)),
        "project": str(Path(yolo["dataset_dir"]) / "runs"),
        "name": active.get("run_name", "mouse_detector_active"),
        "exist_ok": True,
        "seed": int(yolo.get("split_seed", 42)),
        "plots": True,
        "resume": False,
    }
    kwargs.update(base.original._yolo_advanced(cfg, "fine_tune_yolo"))
    print(f"Initializing active-learning fine-tune from: {checkpoint}", flush=True)
    result = YOLO(str(checkpoint)).train(**kwargs)
    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Fine-tuned YOLO checkpoint: {best}", flush=True)
    emit(yolo_trained_model=str(best.resolve()))


ALL_JOBS = dict(base.ALL_JOBS)
ALL_JOBS.update({
    "import_labeled_dataset": job_import_labeled_dataset,
    "mine_active_frames": job_mine_active_frames,
    "fine_tune_yolo": job_fine_tune_yolo,
})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(ALL_JOBS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"=== {args.action} ===", flush=True)
    ALL_JOBS[args.action](cfg)
    print(f"=== {args.action} completed ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
