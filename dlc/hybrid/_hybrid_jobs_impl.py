#!/usr/bin/env python3
"""Background task runner for the English YOLO-SR-DLC Workbench."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dlc import jobs as base_jobs
from dlc.hybrid_pipeline import (
    export_yolo_dataset, generate_traditional_boxes, postprocess_hybrid_predictions,
    prepare_hybrid_video,
)


def load_config(path: Path) -> dict[str, Any]:
    cfg = base_jobs.load_settings(path)
    base = path.resolve().parent
    for section, keys in {
        "yolo": ("dataset_dir", "base_model", "trained_model"),
        "super_resolution": ("model_path",),
    }.items():
        values = cfg.setdefault(section, {})
        for key in keys:
            if values.get(key):
                value = Path(values[key]).expanduser()
                values[key] = str(value if value.is_absolute() else (base / value).resolve())
    return cfg


def emit(**values: Any) -> None:
    print("HYBRID_GUI_RESULT " + json.dumps(values, ensure_ascii=False), flush=True)


def job_hybrid_check(cfg: dict) -> None:
    base_jobs.job_check(cfg)
    for name in ("ultralytics", "opencv-contrib-python"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: MISSING")
    method = cfg.get("super_resolution", {}).get("method", "bicubic")
    model = cfg.get("super_resolution", {}).get("model_path", "")
    if method not in {"none", "bicubic"}:
        print(f"SR model {'OK' if Path(model).is_file() else 'MISSING'}: {model}")
    yolo_model = cfg.get("yolo", {}).get("trained_model", "")
    print(f"YOLO trained model {'OK' if Path(yolo_model).is_file() else 'MISSING'}: {yolo_model}")


def job_generate_boxes(cfg: dict) -> None:
    path = generate_traditional_boxes(cfg)
    emit(box_labels=str(path.resolve()))


def job_export_yolo(cfg: dict) -> None:
    path = export_yolo_dataset(cfg)
    emit(yolo_data=str(path.resolve()))


def _yolo_advanced(cfg: dict, action: str) -> dict:
    value = cfg.get("advanced", {}).get(action, {})
    if not isinstance(value, dict):
        raise ValueError(f"advanced.{action} must be a JSON object")
    return value


def job_train_yolo(cfg: dict) -> None:
    from ultralytics import YOLO
    ycfg = cfg["yolo"]
    data = Path(ycfg["dataset_dir"]) / "data.yaml"
    if not data.is_file():
        raise FileNotFoundError("Export the reviewed YOLO dataset first")
    model = YOLO(ycfg.get("base_model", "yolo26n.pt"))
    kwargs = {
        "data": str(data), "epochs": int(ycfg.get("epochs", 100)),
        "imgsz": int(ycfg.get("image_size", 640)), "batch": int(ycfg.get("batch_size", 8)),
        "device": ycfg.get("device", "auto"), "workers": int(ycfg.get("workers", 4)),
        "patience": int(ycfg.get("patience", 30)), "project": str(Path(ycfg["dataset_dir"]) / "runs"),
        "name": ycfg.get("run_name", "mouse_detector"), "exist_ok": True,
        "seed": int(ycfg.get("split_seed", 42)), "plots": True,
    }
    kwargs.update(_yolo_advanced(cfg, "train_yolo"))
    result = model.train(**kwargs)
    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Best YOLO checkpoint: {best}", flush=True)
    emit(yolo_trained_model=str(best.resolve()))


def job_validate_yolo(cfg: dict) -> None:
    from ultralytics import YOLO
    ycfg = cfg["yolo"]
    model = Path(ycfg["trained_model"])
    if not model.is_file():
        raise FileNotFoundError(model)
    kwargs = {
        "data": str(Path(ycfg["dataset_dir"]) / "data.yaml"),
        "imgsz": int(ycfg.get("image_size", 640)), "batch": int(ycfg.get("batch_size", 8)),
        "device": ycfg.get("device", "auto"), "plots": True,
    }
    kwargs.update(_yolo_advanced(cfg, "validate_yolo"))
    metrics = YOLO(str(model)).val(**kwargs)
    print(f"mAP50-95={float(metrics.box.map):.6f}; mAP50={float(metrics.box.map50):.6f}", flush=True)


def job_prepare_hybrid(cfg: dict) -> None:
    manifests = []
    for value in base_jobs.require_videos(cfg):
        manifest = prepare_hybrid_video(cfg, Path(value))
        manifests.append(manifest)
        print(f"Prepared hybrid input: {manifest}", flush=True)
    emit(hybrid_manifests=manifests)


def _manifest(cfg: dict, source_video: Path) -> tuple[Path, dict]:
    path = Path(cfg["output_dir"]) / "hybrid" / source_video.stem / "hybrid_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prepare the YOLO/SR input first: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def job_run_hybrid_dlc(cfg: dict) -> None:
    from deeplabcut.modelzoo.video_inference import video_inference_superanimal
    model_cfg = cfg.get("model", {})
    for value in base_jobs.require_videos(cfg):
        source = Path(value)
        manifest_path, manifest = _manifest(cfg, source)
        output = manifest_path.parent
        kwargs = {
            "videos": [manifest["dlc_input_video"]],
            "superanimal_name": model_cfg.get("superanimal_name", "superanimal_topviewmouse"),
            "model_name": model_cfg.get("model_name", "hrnet_w32"),
            "detector_name": model_cfg.get("detector_name", "fasterrcnn_resnet50_fpn_v2"),
            "video_adapt": bool(model_cfg.get("video_adapt", True)),
            "max_individuals": 1, "batch_size": int(model_cfg.get("batch_size", 4)),
            "detector_batch_size": int(model_cfg.get("detector_batch_size", 1)),
            "pcutoff": float(model_cfg.get("inference_pcutoff", 0.1)),
            "pseudo_threshold": float(model_cfg.get("pseudo_threshold", 0.1)),
            "bbox_threshold": float(model_cfg.get("bbox_threshold", 0.9)),
            "detector_epochs": int(model_cfg.get("detector_epochs", 1)),
            "pose_epochs": int(model_cfg.get("pose_epochs", 1)),
            "video_adapt_batch_size": int(model_cfg.get("video_adapt_batch_size", 4)),
            "device": model_cfg.get("device", "auto"), "dest_folder": str(output),
            "create_labeled_video": True, "plot_bboxes": True,
        }
        kwargs.update(_yolo_advanced(cfg, "hybrid_dlc"))
        video_inference_superanimal(**kwargs)


def _find_crop_predictions(folder: Path) -> Path:
    candidates = [p for p in folder.glob("dlc_input*.h5") if "_3d" not in p.stem]
    if not candidates:
        raise FileNotFoundError(f"No DLC crop prediction HDF5 in {folder}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def job_postprocess_hybrid(cfg: dict) -> None:
    post = cfg.get("postprocess", {})
    for value in base_jobs.require_videos(cfg):
        source = Path(value)
        manifest_path, manifest = _manifest(cfg, source)
        folder = manifest_path.parent
        csv_path = postprocess_hybrid_predictions(
            predictions_h5=_find_crop_predictions(folder),
            transforms_csv=Path(manifest["transforms"]), source_video=source,
            roi_json=Path(cfg["roi_json"]), output_dir=folder / "final",
            width_cm=float(cfg.get("arena_width_cm", 25)), height_cm=float(cfg.get("arena_height_cm", 30)),
            pcutoff=float(post.get("pcutoff", 0.35)), max_gap_sec=float(post.get("max_gap_sec", 0.2)),
            median_window=int(post.get("median_window", 3)), make_overlay=bool(post.get("write_overlay", True)),
            draw_all_keypoints=bool(post.get("draw_all_keypoints", True)),
        )
        print(f"Final hybrid trajectory: {csv_path}", flush=True)


def job_full_hybrid(cfg: dict) -> None:
    job_prepare_hybrid(cfg)
    job_run_hybrid_dlc(cfg)
    job_postprocess_hybrid(cfg)


HYBRID_JOBS: dict[str, Callable[[dict], None]] = {
    "hybrid_check": job_hybrid_check,
    "generate_boxes": job_generate_boxes,
    "export_yolo": job_export_yolo,
    "train_yolo": job_train_yolo,
    "validate_yolo": job_validate_yolo,
    "prepare_hybrid": job_prepare_hybrid,
    "run_hybrid_dlc": job_run_hybrid_dlc,
    "postprocess_hybrid": job_postprocess_hybrid,
    "full_hybrid": job_full_hybrid,
}

ALL_JOBS = {**base_jobs.JOBS, **HYBRID_JOBS}


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
