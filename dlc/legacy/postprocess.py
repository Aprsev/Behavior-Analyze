#!/usr/bin/env python3
"""Convert SuperAnimal-TopViewMouse predictions to project trajectories.

The module deliberately has no DeepLabCut import, so its coordinate fusion and
tests can run on a CPU-only development machine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


HEAD_WEIGHTS = {
    "head_midpoint": 1.30,
    "nose": 1.15,
    "left_eye": 0.90,
    "right_eye": 0.90,
    "left_ear": 0.65,
    "right_ear": 0.65,
    "left_ear_tip": 0.45,
    "right_ear_tip": 0.45,
    "neck": 0.25,
}

BODY_WEIGHTS = {
    "mouse_center": 3.00,
    "mid_back": 1.20,
    "mid_backend": 1.00,
    "mid_backend2": 0.90,
    "mid_backend3": 0.75,
    "left_shoulder": 0.65,
    "right_shoulder": 0.65,
    "left_midside": 0.85,
    "right_midside": 0.85,
    "left_hip": 0.70,
    "right_hip": 0.70,
    "neck": 0.35,
    "tail_base": 0.25,
}


@dataclass(frozen=True)
class PoseTable:
    """Normalized arrays for a single detected individual."""

    points: dict[str, np.ndarray]  # bodypart -> (frames, 3): x, y, likelihood
    individual: str
    frame_count: int


def _coord_name(value: object) -> str:
    value = str(value).lower()
    return "likelihood" if value in {"score", "confidence", "probability", "p"} else value


def normalize_dlc_dataframe(df: pd.DataFrame) -> PoseTable:
    """Normalize DLC 3- or 4-level HDF columns and select the best individual.

    SuperAnimal output has changed slightly across DLC releases. This parser
    locates levels by their values instead of depending on fixed level numbers.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DLC prediction HDF must have MultiIndex columns")

    levels = [set(map(str, df.columns.get_level_values(i))) for i in range(df.columns.nlevels)]
    coord_level = next((i for i, vals in enumerate(levels) if {"x", "y"}.issubset({v.lower() for v in vals})), None)
    if coord_level is None:
        raise ValueError("Could not find x/y coordinate level in DLC columns")

    known_parts = set(HEAD_WEIGHTS) | set(BODY_WEIGHTS) | {"tail1", "tail2", "tail3", "tail4", "tail5", "tail_end"}
    body_level = max(
        (i for i in range(df.columns.nlevels) if i != coord_level),
        key=lambda i: len({v.lower() for v in levels[i]} & known_parts),
    )
    if not ({v.lower() for v in levels[body_level]} & known_parts):
        raise ValueError("Could not find TopViewMouse bodyparts in DLC columns")

    remaining = [i for i in range(df.columns.nlevels) if i not in {coord_level, body_level}]
    individual_level = None
    for i in remaining:
        vals = levels[i]
        # Scorer levels normally contain one long DLC model name. An individual
        # level is named as such or has several compact identifiers.
        level_name = str(df.columns.names[i] or "").lower()
        if "individual" in level_name or len(vals) > 1:
            individual_level = i
            break

    individuals = [""] if individual_level is None else list(dict.fromkeys(map(str, df.columns.get_level_values(individual_level))))
    candidates: dict[str, dict[str, np.ndarray]] = {}
    quality: dict[str, float] = {}
    for individual in individuals:
        parts: dict[str, dict[str, np.ndarray]] = {}
        for col in df.columns:
            if individual_level is not None and str(col[individual_level]) != individual:
                continue
            bp = str(col[body_level]).lower()
            coord = _coord_name(col[coord_level])
            if coord not in {"x", "y", "likelihood"}:
                continue
            parts.setdefault(bp, {})[coord] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        arrays: dict[str, np.ndarray] = {}
        for bp, coords in parts.items():
            if "x" not in coords or "y" not in coords:
                continue
            likelihood = coords.get("likelihood", np.ones(len(df), dtype=float))
            arrays[bp] = np.column_stack([coords["x"], coords["y"], likelihood])
        candidates[individual] = arrays
        torso_scores = [a[:, 2] for bp, a in arrays.items() if bp in BODY_WEIGHTS]
        quality[individual] = float(np.nanmedian(np.column_stack(torso_scores))) if torso_scores else -np.inf

    selected = max(individuals, key=lambda name: quality[name])
    return PoseTable(candidates[selected], selected, len(df))


def _weighted_fusion(table: PoseTable, weights: dict[str, float], pcutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted keypoint fusion with per-frame spatial rejection."""
    n = table.frame_count
    xy = np.full((n, 2), np.nan, dtype=float)
    confidence = np.full(n, np.nan, dtype=float)
    available = [(bp, table.points[bp], w) for bp, w in weights.items() if bp in table.points]
    for frame in range(n):
        candidates = []
        for _bp, arr, base_weight in available:
            x, y, likelihood = arr[frame]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(likelihood) and likelihood >= pcutoff:
                candidates.append((x, y, likelihood, base_weight))
        if not candidates:
            continue
        pts = np.asarray([[c[0], c[1]] for c in candidates], dtype=float)
        # Reject isolated hallucinations when at least three related landmarks
        # survive. Threshold is adaptive to the visible head/body scale.
        if len(pts) >= 3:
            center = np.median(pts, axis=0)
            distance = np.linalg.norm(pts - center, axis=1)
            med = float(np.median(distance))
            keep = distance <= max(12.0, 2.5 * med)
        else:
            keep = np.ones(len(pts), dtype=bool)
        kept = [c for c, use in zip(candidates, keep) if use]
        if not kept:
            continue
        effective = np.asarray([c[2] * c[3] for c in kept])
        kept_xy = np.asarray([[c[0], c[1]] for c in kept])
        xy[frame] = np.average(kept_xy, axis=0, weights=effective)
        confidence[frame] = np.average([c[2] for c in kept], weights=[c[3] for c in kept])
    return xy, confidence


def fuse_pose(table: PoseTable, pcutoff: float) -> dict[str, np.ndarray]:
    head, head_conf = _weighted_fusion(table, HEAD_WEIGHTS, pcutoff)
    # ``mouse_center`` is a semantic keypoint in SuperAnimal-TopViewMouse and
    # is therefore the closest available definition of the requested body
    # centroid. Other torso landmarks are an occlusion fallback, not points
    # that should pull a visible mouse_center away from its prediction.
    fallback_weights = {bp: weight for bp, weight in BODY_WEIGHTS.items() if bp != "mouse_center"}
    body, body_conf = _weighted_fusion(table, fallback_weights, pcutoff)
    if "mouse_center" in table.points:
        center = table.points["mouse_center"]
        use_center = np.isfinite(center).all(axis=1) & (center[:, 2] >= pcutoff)
        body[use_center] = center[use_center, :2]
        body_conf[use_center] = center[use_center, 2]
    return {"head": head, "head_confidence": head_conf, "body": body, "body_confidence": body_conf}


def _fill_short_gaps(values: np.ndarray, max_gap_frames: int, median_window: int) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    # pandas ``limit`` can partially fill a long gap. Fill explicit bounded
    # runs instead, so an occlusion longer than the configured limit remains
    # entirely missing rather than looking like a short valid segment.
    valid = np.isfinite(result).all(axis=1)
    if max_gap_frames > 0 and len(result) > 2:
        missing = np.flatnonzero(~valid)
        for run in np.split(missing, np.where(np.diff(missing) != 1)[0] + 1):
            if not len(run):
                continue
            start, end = int(run[0]), int(run[-1])
            if len(run) <= max_gap_frames and start > 0 and end + 1 < len(result) and valid[start - 1] and valid[end + 1]:
                for dim in range(result.shape[1]):
                    result[run, dim] = np.interp(run, [start - 1, end + 1], [result[start - 1, dim], result[end + 1, dim]])
    valid_after_fill = np.isfinite(result).all(axis=1)
    if median_window > 1:
        result = pd.DataFrame(result).rolling(median_window, center=True, min_periods=1).median().to_numpy(float)
        result[~valid_after_fill] = np.nan
    return result


def load_arena_transform(roi_json: Path, width_cm: float, height_cm: float) -> np.ndarray:
    data = json.loads(roi_json.read_text(encoding="utf-8"))
    corners = np.asarray(data["arena_corners_px"], dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError("roi_json arena_corners_px must contain TL, TR, BR, BL")
    target = np.asarray([[0, 0], [width_cm, 0], [width_cm, height_cm], [0, height_cm]], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, target)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    out = np.full_like(points, np.nan, dtype=float)
    valid = np.isfinite(points).all(axis=1)
    if valid.any():
        out[valid] = cv2.perspectiveTransform(points[valid].astype(np.float32)[None], matrix)[0]
    return out


def _all_keypoints_for_frame(table: PoseTable, frame: int, pcutoff: float) -> Iterable[tuple[str, int, int, float]]:
    for name, arr in table.points.items():
        x, y, likelihood = arr[frame]
        if np.isfinite(x) and np.isfinite(y) and likelihood >= pcutoff:
            yield name, int(round(x)), int(round(y)), float(likelihood)


def write_overlay(video: Path, output: Path, trajectory: pd.DataFrame, table: PoseTable, pcutoff: float, draw_all: bool) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create overlay: {output}")
    frame = 0
    while frame < len(trajectory):
        ok, image = cap.read()
        if not ok:
            break
        if draw_all:
            for _name, x, y, likelihood in _all_keypoints_for_frame(table, frame, pcutoff):
                cv2.circle(image, (x, y), 2, (0, int(100 + 155 * likelihood), 255), -1, cv2.LINE_AA)
        row = trajectory.iloc[frame]
        if np.isfinite(row.body_x_px):
            cv2.circle(image, (int(round(row.body_x_px)), int(round(row.body_y_px))), 6, (0, 255, 0), -1, cv2.LINE_AA)
        if np.isfinite(row.head_x_px):
            cv2.circle(image, (int(round(row.head_x_px)), int(round(row.head_y_px))), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(image, f"frame {frame}  green=body red=head", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(image)
        frame += 1
    writer.release()
    cap.release()


def postprocess_predictions(
    predictions_h5: Path,
    video: Path,
    roi_json: Path,
    output_dir: Path,
    width_cm: float,
    height_cm: float,
    pcutoff: float = 0.35,
    max_gap_sec: float = 0.2,
    median_window: int = 3,
    make_overlay: bool = True,
    draw_all_keypoints: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_hdf(predictions_h5)
    table = normalize_dlc_dataframe(raw)
    fused = fuse_pose(table, pcutoff)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS is missing or invalid")
    if video_frames > 0 and abs(video_frames - table.frame_count) > 1:
        raise ValueError(f"Prediction/video frame mismatch: {table.frame_count} vs {video_frames}")

    max_gap_frames = max(0, int(round(max_gap_sec * fps)))
    body_raw, head_raw = fused["body"], fused["head"]
    body = _fill_short_gaps(body_raw, max_gap_frames, median_window)
    head = _fill_short_gaps(head_raw, max_gap_frames, median_window)
    homography = load_arena_transform(roi_json, width_cm, height_cm)
    body_cm, head_cm = transform_points(body, homography), transform_points(head, homography)

    n = table.frame_count
    trajectory = pd.DataFrame({
        "frame": np.arange(n, dtype=int),
        "timestamp_sec": np.arange(n, dtype=float) / fps,
        "body_x_cm": body_cm[:, 0], "body_y_cm": body_cm[:, 1],
        "head_x_cm": head_cm[:, 0], "head_y_cm": head_cm[:, 1],
        "body_x_px": body[:, 0], "body_y_px": body[:, 1],
        "head_x_px": head[:, 0], "head_y_px": head[:, 1],
        "body_x_px_raw": body_raw[:, 0], "body_y_px_raw": body_raw[:, 1],
        "head_x_px_raw": head_raw[:, 0], "head_y_px_raw": head_raw[:, 1],
        "body_confidence": fused["body_confidence"],
        "head_confidence": fused["head_confidence"],
        "body_interpolated": np.isfinite(body[:, 0]) & ~np.isfinite(body_raw[:, 0]),
        "head_interpolated": np.isfinite(head[:, 0]) & ~np.isfinite(head_raw[:, 0]),
    })
    csv_path = output_dir / "trajectory.csv"
    trajectory.to_csv(csv_path, index=False, float_format="%.5f")

    raw.to_hdf(output_dir / "dlc_keypoints.h5", key="df_with_missing", mode="w")
    raw.to_csv(output_dir / "dlc_keypoints.csv")
    qa = {
        "source_video": str(video.resolve()),
        "source_predictions": str(predictions_h5.resolve()),
        "selected_individual": table.individual,
        "frames": n,
        "fps": fps,
        "pcutoff": pcutoff,
        "max_gap_sec": max_gap_sec,
        "body_raw_valid_percent": round(100 * float(np.isfinite(body_raw[:, 0]).mean()), 3),
        "head_raw_valid_percent": round(100 * float(np.isfinite(head_raw[:, 0]).mean()), 3),
        "body_final_valid_percent": round(100 * float(np.isfinite(body[:, 0]).mean()), 3),
        "head_final_valid_percent": round(100 * float(np.isfinite(head[:, 0]).mean()), 3),
        "bodyparts_found": sorted(table.points),
    }
    (output_dir / "quality_report.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    if make_overlay:
        write_overlay(video, output_dir / "annotated_output.mp4", trajectory, table, pcutoff, draw_all_keypoints)
    return csv_path
