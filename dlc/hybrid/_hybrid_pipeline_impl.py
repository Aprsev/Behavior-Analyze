#!/usr/bin/env python3
"""YOLO -> super-resolution -> DeepLabCut hybrid mouse pipeline.

This module contains CPU-testable dataset, crop-transform, and post-processing
code. Heavy Ultralytics and DeepLabCut imports are intentionally lazy.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRADITIONAL_CODE = ROOT / "traditional" / "code"

BOX_COLUMNS = [
    "video", "frame", "image", "x1", "y1", "x2", "y2", "source",
    "confidence", "exclude", "reviewed",
]


def _video_properties(video: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {video}")
    return count, fps, width, height


def _load_roi(roi_json: Path) -> np.ndarray:
    data = json.loads(roi_json.read_text(encoding="utf-8"))
    corners = np.asarray(data["arena_corners_px"], np.float32)
    if corners.shape != (4, 2):
        raise ValueError("arena_corners_px must be TL, TR, BR, BL")
    return corners


def _perspective(corners: np.ndarray, width_cm: float, height_cm: float):
    out_w = max(100, int(round(width_cm * 10)))
    out_h = max(100, int(round(height_cm * 10)))
    dst = np.asarray([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    return out_w, out_h, cv2.getPerspectiveTransform(corners, dst), cv2.getPerspectiveTransform(dst, corners)


def _read_frames(video: Path, indices: np.ndarray) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    frames: list[tuple[int, np.ndarray]] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append((int(index), frame))
    cap.release()
    return frames


def _traditional_functions():
    if str(TRADITIONAL_CODE) not in sys.path:
        sys.path.insert(0, str(TRADITIONAL_CODE))
    from mouse_behavior_pipeline import robust_threshold, segment_mouse
    return robust_threshold, segment_mouse


def _bbox_from_rectified_detection(
    bbox: tuple[int, int, int, int], inverse: np.ndarray,
    frame_width: int, frame_height: int, padding: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    rect = np.asarray([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], np.float32)
    source = cv2.perspectiveTransform(rect, inverse)[0]
    x1, y1 = source.min(axis=0)
    x2, y2 = source.max(axis=0)
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    return (
        max(0, int(math.floor(x1 - pad_x))), max(0, int(math.floor(y1 - pad_y))),
        min(frame_width - 1, int(math.ceil(x2 + pad_x))),
        min(frame_height - 1, int(math.ceil(y2 + pad_y))),
    )


def generate_traditional_boxes(cfg: dict[str, Any]) -> Path:
    """Generate auditable YOLO pseudo-label boxes from background subtraction."""
    ycfg = cfg.get("yolo", {})
    dataset = Path(ycfg["dataset_dir"]).resolve()
    source_images = dataset / "source_images"
    source_images.mkdir(parents=True, exist_ok=True)
    roi = _load_roi(Path(cfg["roi_json"]))
    out_w, out_h, forward, inverse = _perspective(
        roi, float(cfg.get("arena_width_cm", 25)), float(cfg.get("arena_height_cm", 30))
    )
    robust_threshold, segment_mouse = _traditional_functions()
    rows: list[dict[str, Any]] = []
    per_video = int(ycfg.get("samples_per_video", 80))
    bg_samples = int(ycfg.get("background_samples", 61))
    padding = float(ycfg.get("auto_box_padding", 0.15))
    percentile = float(ycfg.get("background_percentile", 85))
    requested_threshold = float(ycfg.get("background_threshold", 0))

    for video_value in cfg["videos"]:
        video = Path(video_value)
        count, _fps, width, height = _video_properties(video)
        calibration_indices = np.unique(np.linspace(0, count - 1, min(bg_samples, count), dtype=int))
        calibration = _read_frames(video, calibration_indices)
        if len(calibration) < 3:
            raise RuntimeError(f"Not enough frames for background model: {video}")
        rectified_samples = np.stack([cv2.warpPerspective(frame, forward, (out_w, out_h)) for _, frame in calibration])
        background = np.percentile(rectified_samples, percentile, axis=0).astype(np.uint8)
        threshold, stats = robust_threshold(rectified_samples, background, requested_threshold)
        print(f"{video.name}: background threshold={threshold:.2f}; stats={stats}", flush=True)

        indices = np.unique(np.linspace(0, count - 1, min(per_video, count), dtype=int))
        for frame_index, source_frame in _read_frames(video, indices):
            rectified = cv2.warpPerspective(source_frame, forward, (out_w, out_h))
            detection = segment_mouse(rectified, background, threshold, None)
            image_name = f"{video.stem.replace(' ', '_')}_{frame_index:08d}.jpg"
            image_path = source_images / image_name
            cv2.imwrite(str(image_path), source_frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
            if detection.bbox is None:
                rows.append({
                    "video": str(video.resolve()), "frame": frame_index, "image": str(image_path.resolve()),
                    "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan,
                    "source": "traditional_missing", "confidence": 0.0, "exclude": True, "reviewed": False,
                })
                continue
            x1, y1, x2, y2 = _bbox_from_rectified_detection(
                detection.bbox, inverse, width, height, padding
            )
            box_area = max(1, (x2 - x1) * (y2 - y1))
            compactness = min(1.0, float(detection.area) / max(box_area, 1))
            confidence = float(np.clip(0.35 + 1.8 * compactness, 0.05, 0.95))
            rows.append({
                "video": str(video.resolve()), "frame": frame_index, "image": str(image_path.resolve()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "source": "traditional_background", "confidence": confidence,
                "exclude": False, "reviewed": False,
            })
    labels_csv = dataset / "box_labels.csv"
    pd.DataFrame(rows, columns=BOX_COLUMNS).to_csv(labels_csv, index=False, float_format="%.5f")
    print(f"Wrote {len(rows)} pseudo-labels to {labels_csv}", flush=True)
    return labels_csv


def load_box_labels(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    for column in BOX_COLUMNS:
        if column not in rows:
            rows[column] = False if column in {"exclude", "reviewed"} else np.nan
    for column in ("exclude", "reviewed"):
        rows[column] = rows[column].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return rows


def save_box_labels(path: Path, rows: pd.DataFrame) -> None:
    temp = path.with_suffix(".tmp.csv")
    rows.to_csv(temp, index=False, float_format="%.5f")
    temp.replace(path)


def update_box_label(path: Path, row_index: int, box: tuple[float, float, float, float] | None, exclude: bool) -> None:
    rows = load_box_labels(path)
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(row_index)
    if box is not None:
        rows.loc[row_index, ["x1", "y1", "x2", "y2"]] = list(box)
        rows.loc[row_index, "source"] = "manual_review"
        rows.loc[row_index, "confidence"] = 1.0
    rows.loc[row_index, "exclude"] = bool(exclude)
    rows.loc[row_index, "reviewed"] = True
    save_box_labels(path, rows)


def export_yolo_dataset(cfg: dict[str, Any]) -> Path:
    """Export reviewed CSV labels to the official normalized YOLO format."""
    ycfg = cfg.get("yolo", {})
    root = Path(ycfg["dataset_dir"]).resolve()
    rows = load_box_labels(root / "box_labels.csv")
    valid = rows.loc[
        ~rows.exclude & rows[["x1", "y1", "x2", "y2"]].notna().all(axis=1)
    ].copy()
    if len(valid) < 2:
        raise ValueError("At least two non-excluded reviewed/auto boxes are required")
    val_fraction = float(ycfg.get("validation_fraction", 0.2))
    seed = int(ycfg.get("split_seed", 42))
    assignments = []
    for _, row in valid.iterrows():
        token = f"{seed}:{row.video}:{int(row.frame)}".encode()
        fraction = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
        assignments.append("val" if fraction < val_fraction else "train")
    valid["split"] = assignments
    if not (valid.split == "val").any():
        valid.iloc[-1, valid.columns.get_loc("split")] = "val"
    if not (valid.split == "train").any():
        valid.iloc[0, valid.columns.get_loc("split")] = "train"

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    for _, row in valid.iterrows():
        source = Path(row.image)
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Cannot read dataset image: {source}")
        height, width = image.shape[:2]
        x1, y1 = np.clip([float(row.x1), float(row.y1)], [0, 0], [width - 1, height - 1])
        x2, y2 = np.clip([float(row.x2), float(row.y2)], [0, 0], [width - 1, height - 1])
        if x2 <= x1 or y2 <= y1:
            continue
        destination = root / "images" / row.split / source.name
        shutil.copy2(source, destination)
        xc, yc = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
        bw, bh = ((x2 - x1) / width, (y2 - y1) / height)
        (root / "labels" / row.split / f"{source.stem}.txt").write_text(
            f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8"
        )
    data_yaml = root / "data.yaml"
    # YAML kept deliberately minimal and dependency-free.
    data_yaml.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: mouse\n",
        encoding="utf-8",
    )
    manifest = {
        "labels_csv": str((root / "box_labels.csv").resolve()),
        "train_count": int((valid.split == "train").sum()),
        "val_count": int((valid.split == "val").sum()),
        "class": "mouse",
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"YOLO dataset: {manifest}", flush=True)
    return data_yaml


class SuperResolver:
    """Pluggable SR stage with explicit no-model fallback modes."""

    def __init__(self, method: str, model_path: str, scale: int):
        self.method = method.lower()
        self.scale = int(scale)
        self.impl = None
        if self.method in {"edsr", "espcn", "fsrcnn", "lapsrn"}:
            if not model_path or not Path(model_path).is_file():
                raise FileNotFoundError(f"{self.method.upper()} model file is required: {model_path}")
            if not hasattr(cv2, "dnn_superres"):
                raise RuntimeError("Install opencv-contrib-python for neural super-resolution")
            self.impl = cv2.dnn_superres.DnnSuperResImpl_create()
            self.impl.readModel(str(model_path))
            self.impl.setModel(self.method, self.scale)
        elif self.method not in {"bicubic", "none"}:
            raise ValueError(f"Unknown super-resolution method: {method}")

    def apply(self, crop: np.ndarray, output_size: int) -> np.ndarray:
        if self.impl is not None:
            crop = self.impl.upsample(crop)
        interpolation = cv2.INTER_CUBIC if self.method != "none" else cv2.INTER_LINEAR
        return cv2.resize(crop, (output_size, output_size), interpolation=interpolation)


def _square_crop(frame: np.ndarray, box: np.ndarray, scale: float) -> tuple[np.ndarray, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    size = max(x2 - x1, y2 - y1) * scale
    size = max(size, 8.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0, y0 = cx - size / 2, cy - size / 2
    integer_size = max(8, int(math.ceil(size)))
    matrix = np.asarray([[1, 0, -x0], [0, 1, -y0]], np.float32)
    crop = cv2.warpAffine(
        frame, matrix, (integer_size, integer_size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return crop, x0, y0, float(integer_size)


def prepare_hybrid_video(cfg: dict[str, Any], video: Path) -> dict[str, str]:
    """Detect the mouse, create SR crops, and persist inverse transforms."""
    from ultralytics import YOLO

    ycfg, srcfg = cfg.get("yolo", {}), cfg.get("super_resolution", {})
    model_path = Path(ycfg["trained_model"])
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")
    output_root = Path(cfg["output_dir"]).resolve() / "hybrid" / video.stem
    output_root.mkdir(parents=True, exist_ok=True)
    count, fps, width, height = _video_properties(video)
    output_size = int(srcfg.get("output_size", 512))
    if output_size < 64:
        raise ValueError("super_resolution.output_size must be at least 64")
    resolver = SuperResolver(
        srcfg.get("method", "bicubic"), srcfg.get("model_path", ""), int(srcfg.get("scale", 4))
    )
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video))
    crop_video = output_root / "dlc_input.mp4"
    overlay_video = output_root / "yolo_detection.mp4"
    crop_writer = cv2.VideoWriter(str(crop_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_size, output_size))
    overlay_writer = cv2.VideoWriter(str(overlay_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not crop_writer.isOpened() or not overlay_writer.isOpened():
        raise RuntimeError("Cannot create hybrid intermediate videos")
    previous_box: np.ndarray | None = None
    lost = 0
    max_fallback = max(0, int(round(float(ycfg.get("max_fallback_sec", 0.3)) * fps)))
    rows = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(
            frame, conf=float(ycfg.get("confidence", 0.25)), iou=float(ycfg.get("iou", 0.7)),
            imgsz=int(ycfg.get("image_size", 640)), device=ycfg.get("device", "auto"), verbose=False,
        )[0]
        box = None
        confidence = np.nan
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            valid = np.flatnonzero(classes == 0)
            if len(valid):
                chosen = valid[int(np.argmax(confs[valid]))]
                box, confidence = xyxy[chosen], float(confs[chosen])
                previous_box, lost = box.copy(), 0
                source = "yolo"
        if box is None and previous_box is not None and lost < max_fallback:
            lost += 1
            box, source = previous_box.copy(), "temporal_fallback"
        elif box is None:
            lost += 1
            source = "missing"

        shown = frame.copy()
        if box is None:
            crop = np.zeros((output_size, output_size, 3), np.uint8)
            x0 = y0 = crop_size = np.nan
            cv2.putText(shown, "YOLO: MISSING", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        else:
            raw_crop, x0, y0, crop_size = _square_crop(frame, box, float(ycfg.get("crop_scale", 1.5)))
            crop = resolver.apply(raw_crop, output_size)
            bx = np.rint(box).astype(int)
            color = (0, 255, 0) if source == "yolo" else (0, 180, 255)
            cv2.rectangle(shown, tuple(bx[:2]), tuple(bx[2:]), color, 2)
            cv2.putText(shown, f"{source} {confidence:.3f}" if np.isfinite(confidence) else source,
                        (max(0, bx[0]), max(20, bx[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        crop_writer.write(crop)
        overlay_writer.write(shown)
        rows.append({
            "frame": frame_index, "x0": x0, "y0": y0, "crop_size": crop_size,
            "output_size": output_size, "box_x1": np.nan if box is None else box[0],
            "box_y1": np.nan if box is None else box[1], "box_x2": np.nan if box is None else box[2],
            "box_y2": np.nan if box is None else box[3], "detector_confidence": confidence,
            "box_source": source,
        })
        frame_index += 1
        if frame_index % 250 == 0:
            print(f"Prepared {frame_index}/{count} frames", flush=True)
    cap.release(); crop_writer.release(); overlay_writer.release()
    transforms = output_root / "crop_transforms.csv"
    pd.DataFrame(rows).to_csv(transforms, index=False, float_format="%.6f")
    manifest = {
        "source_video": str(video.resolve()), "dlc_input_video": str(crop_video.resolve()),
        "transforms": str(transforms.resolve()), "yolo_overlay": str(overlay_video.resolve()),
        "frames": frame_index, "super_resolution": srcfg,
    }
    manifest_path = output_root / "hybrid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {key: str(value) for key, value in manifest.items() if isinstance(value, str)}


def map_pose_table_to_source(table, transforms_csv: Path):
    """Map every DLC crop keypoint back to the original video coordinates."""
    transforms = pd.read_csv(transforms_csv)
    if len(transforms) != table.frame_count:
        raise ValueError(f"Transform/prediction frame mismatch: {len(transforms)} vs {table.frame_count}")
    x0 = pd.to_numeric(transforms.x0, errors="coerce").to_numpy(float)
    y0 = pd.to_numeric(transforms.y0, errors="coerce").to_numpy(float)
    size = pd.to_numeric(transforms.crop_size, errors="coerce").to_numpy(float)
    output_size = pd.to_numeric(transforms.output_size, errors="coerce").to_numpy(float)
    mapped = {}
    for name, values in table.points.items():
        values = values.copy()
        valid = np.isfinite(values[:, :2]).all(axis=1) & np.isfinite(x0 + y0 + size + output_size) & (output_size > 0)
        values[valid, 0] = x0[valid] + values[valid, 0] * size[valid] / output_size[valid]
        values[valid, 1] = y0[valid] + values[valid, 1] * size[valid] / output_size[valid]
        values[~valid, :2] = np.nan
        mapped[name] = values
    return replace(table, points=mapped), transforms


def postprocess_hybrid_predictions(
    predictions_h5: Path, transforms_csv: Path, source_video: Path, roi_json: Path,
    output_dir: Path, width_cm: float, height_cm: float, pcutoff: float,
    max_gap_sec: float, median_window: int, make_overlay: bool, draw_all_keypoints: bool,
) -> Path:
    """Fuse DLC crop predictions after exact inverse mapping to source pixels."""
    from dlc.postprocess import (
        _fill_short_gaps, fuse_pose, load_arena_transform, normalize_dlc_dataframe,
        transform_points, write_overlay,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_hdf(predictions_h5)
    crop_table = normalize_dlc_dataframe(raw)
    table, transforms = map_pose_table_to_source(crop_table, transforms_csv)
    fused = fuse_pose(table, pcutoff)
    count, fps, _width, _height = _video_properties(source_video)
    if abs(count - table.frame_count) > 1:
        raise ValueError(f"Source/prediction frame mismatch: {count} vs {table.frame_count}")
    gap = max(0, int(round(max_gap_sec * fps)))
    body_raw, head_raw = fused["body"], fused["head"]
    body = _fill_short_gaps(body_raw, gap, median_window)
    head = _fill_short_gaps(head_raw, gap, median_window)
    homography = load_arena_transform(roi_json, width_cm, height_cm)
    body_cm, head_cm = transform_points(body, homography), transform_points(head, homography)
    n = table.frame_count
    trajectory = pd.DataFrame({
        "frame": np.arange(n), "timestamp_sec": np.arange(n) / fps,
        "body_x_cm": body_cm[:, 0], "body_y_cm": body_cm[:, 1],
        "head_x_cm": head_cm[:, 0], "head_y_cm": head_cm[:, 1],
        "body_x_px": body[:, 0], "body_y_px": body[:, 1],
        "head_x_px": head[:, 0], "head_y_px": head[:, 1],
        "body_x_px_raw": body_raw[:, 0], "body_y_px_raw": body_raw[:, 1],
        "head_x_px_raw": head_raw[:, 0], "head_y_px_raw": head_raw[:, 1],
        "body_confidence": fused["body_confidence"], "head_confidence": fused["head_confidence"],
        "body_interpolated": np.isfinite(body[:, 0]) & ~np.isfinite(body_raw[:, 0]),
        "head_interpolated": np.isfinite(head[:, 0]) & ~np.isfinite(head_raw[:, 0]),
        "yolo_confidence": transforms.detector_confidence,
        "yolo_box_source": transforms.box_source,
    })
    csv_path = output_dir / "trajectory.csv"
    trajectory.to_csv(csv_path, index=False, float_format="%.5f")
    # Preserve both raw crop-space DLC output and mapped source-space points.
    raw.to_hdf(output_dir / "dlc_crop_keypoints.h5", key="df_with_missing", mode="w")
    mapped_columns = {}
    for name, values in table.points.items():
        mapped_columns[(name, "x")] = values[:, 0]
        mapped_columns[(name, "y")] = values[:, 1]
        mapped_columns[(name, "likelihood")] = values[:, 2]
    pd.DataFrame(mapped_columns).to_csv(output_dir / "dlc_source_keypoints.csv", index=False)
    qa = {
        "source_video": str(source_video.resolve()), "crop_predictions": str(predictions_h5.resolve()),
        "crop_transforms": str(transforms_csv.resolve()), "frames": n, "fps": fps,
        "yolo_direct_percent": round(100 * float((transforms.box_source == "yolo").mean()), 3),
        "yolo_fallback_percent": round(100 * float((transforms.box_source == "temporal_fallback").mean()), 3),
        "yolo_missing_percent": round(100 * float((transforms.box_source == "missing").mean()), 3),
        "body_valid_percent": round(100 * float(np.isfinite(body[:, 0]).mean()), 3),
        "head_valid_percent": round(100 * float(np.isfinite(head[:, 0]).mean()), 3),
        "bodyparts_found": sorted(table.points),
    }
    (output_dir / "quality_report.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if make_overlay:
        write_overlay(source_video, output_dir / "annotated_output.mp4", trajectory, table, pcutoff, draw_all_keypoints)
    return csv_path
