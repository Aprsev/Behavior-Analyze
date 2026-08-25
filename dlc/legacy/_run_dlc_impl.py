#!/usr/bin/env python3
"""Run zero-shot/adapted SuperAnimal inference and project post-processing."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dlc.postprocess import postprocess_predictions


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    for key in ("video", "roi_json", "output_dir"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = value if value.is_absolute() else (base / value).resolve()
    return cfg


def check(cfg: dict) -> int:
    print(f"Python: {sys.version.split()[0]}")
    for package in ("deeplabcut", "torch", "opencv-python", "pandas", "tables"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: MISSING")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    except ImportError:
        print("CUDA available: unknown (PyTorch missing)")
    print(f"Video: {'OK' if cfg['video'].is_file() else 'MISSING'}  {cfg['video']}")
    print(f"ROI: {'OK' if cfg['roi_json'].is_file() else 'MISSING'}  {cfg['roi_json']}")
    ffmpeg = shutil.which("ffmpeg")
    print(f"FFmpeg: {ffmpeg or 'not on PATH (OpenCV may still decode the video)'}")
    return 0 if cfg["video"].is_file() and cfg["roi_json"].is_file() else 2


def run_inference(cfg: dict) -> None:
    try:
        from deeplabcut.modelzoo.video_inference import video_inference_superanimal
    except ImportError as exc:
        raise RuntimeError("DeepLabCut Model Zoo is not installed; install dlc/requirements-dlc.txt") from exc

    video, out = cfg["video"], cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    model = cfg.get("model", {})
    adapt = bool(model.get("video_adapt", True))
    kwargs = {
        "videos": [str(video)],
        "superanimal_name": model.get("superanimal_name", "superanimal_topviewmouse"),
        "model_name": model.get("model_name", "hrnet_w32"),
        "detector_name": model.get("detector_name", "fasterrcnn_resnet50_fpn_v2"),
        "video_adapt": adapt,
        "max_individuals": int(model.get("max_individuals", 1)),
        "batch_size": int(model.get("batch_size", 4)),
        "detector_batch_size": int(model.get("detector_batch_size", 1)),
        "dest_folder": str(out),
    }
    if adapt:
        kwargs.update({
            "pseudo_threshold": float(model.get("pseudo_threshold", 0.1)),
            "bbox_threshold": float(model.get("bbox_threshold", 0.9)),
            "detector_epochs": int(model.get("detector_epochs", 1)),
            "pose_epochs": int(model.get("pose_epochs", 1)),
        })
    print("Running DeepLabCut SuperAnimal inference. The first run downloads model weights.")
    video_inference_superanimal(**kwargs)


def find_predictions(cfg: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit.resolve()
    out, stem = cfg["output_dir"], cfg["video"].stem
    candidates = [p for p in out.glob(f"{stem}*.h5") if "_3d" not in p.stem and p.name != "dlc_keypoints.h5"]
    if not candidates:
        candidates = [p for p in out.glob("*.h5") if "_3d" not in p.stem and p.name != "dlc_keypoints.h5"]
    if not candidates:
        raise FileNotFoundError(f"No DeepLabCut prediction HDF5 found in {out}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_postprocess(cfg: dict, predictions: Path | None) -> Path:
    post = cfg.get("postprocess", {})
    h5 = find_predictions(cfg, predictions)
    print(f"Post-processing: {h5}")
    return postprocess_predictions(
        predictions_h5=h5,
        video=cfg["video"],
        roi_json=cfg["roi_json"],
        output_dir=cfg["output_dir"],
        width_cm=float(cfg.get("arena_width_cm", 25.0)),
        height_cm=float(cfg.get("arena_height_cm", 30.0)),
        pcutoff=float(post.get("pcutoff", 0.35)),
        max_gap_sec=float(post.get("max_gap_sec", 0.2)),
        median_window=int(post.get("median_window", 3)),
        make_overlay=bool(post.get("write_overlay", True)),
        draw_all_keypoints=bool(post.get("draw_all_keypoints", True)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "infer", "postprocess", "all"])
    parser.add_argument("--config", type=Path, default=HERE / "config.json", help="JSON config (default: dlc/config.json)")
    parser.add_argument("--predictions", type=Path, help="existing DLC .h5, used by postprocess")
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"Config not found: {args.config}. Copy config.example.json to config.json first.")
    cfg = load_config(args.config)
    if args.mode == "check":
        return check(cfg)
    if not cfg["video"].is_file() or not cfg["roi_json"].is_file():
        raise FileNotFoundError("Configured video or ROI JSON does not exist; run check for details")
    if args.mode in {"infer", "all"}:
        run_inference(cfg)
    if args.mode in {"postprocess", "all"}:
        result = run_postprocess(cfg, args.predictions)
        print(f"Trajectory written: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
