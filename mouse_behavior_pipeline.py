#!/usr/bin/env python3
"""Automated single-mouse open-field tracking from a static-camera MP4.

The default input path is intentionally empty. If --input is omitted, the
script uses the only MP4 in the working directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DEFAULT_INPUT_VIDEO = ""
ARENA_WIDTH_CM = 25.0
ARENA_HEIGHT_CM = 30.0
RECTIFIED_PX_PER_CM = 10.0


class PipelineError(RuntimeError):
    pass


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.lines.append(message)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_VIDEO, help="Input MP4; blank auto-discovers one MP4")
    parser.add_argument("--output-dir", default=".", help="Directory for requested outputs")
    parser.add_argument("--arena-width-cm", type=float, default=ARENA_WIDTH_CM)
    parser.add_argument("--arena-height-cm", type=float, default=ARENA_HEIGHT_CM)
    parser.add_argument("--threshold", type=float, default=0.0, help="Difference threshold; 0 estimates it")
    parser.add_argument("--corners", default="", help="Override TL,TR,BR,BL corners: x,y;x,y;x,y;x,y")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit; 0 processes all frames")
    parser.add_argument("--no-contour", action="store_true", help="Do not draw mouse contour")
    return parser.parse_args()


def resolve_input(value: str, output_dir: Path) -> Path:
    if value.strip():
        path = Path(value).expanduser().resolve()
    else:
        excluded = (output_dir / "annotated_output.mp4").resolve()
        choices = [p.resolve() for p in Path.cwd().glob("*.mp4") if p.resolve() != excluded]
        if len(choices) != 1:
            raise PipelineError(
                f"Input path is blank and found {len(choices)} candidate MP4 files in {Path.cwd()}. "
                "Place exactly one MP4 here or pass --input PATH."
            )
        path = choices[0]
    if not path.is_file() or path.suffix.lower() != ".mp4":
        raise PipelineError(f"Input must be an existing MP4 file: {path}")
    return path


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    pts = np.roll(pts, -start, axis=0)
    # Angle sorting gives TL,TR,BR,BL for image coordinates; repair if reversed.
    if pts[1, 0] < pts[-1, 0]:
        pts = pts[[0, 3, 2, 1]]
    return pts.astype(np.float32)


def parse_corners(value: str) -> np.ndarray:
    try:
        pts = [[float(v) for v in pair.split(",")] for pair in value.split(";")]
    except ValueError as exc:
        raise PipelineError("Invalid --corners. Expected x,y;x,y;x,y;x,y") from exc
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise PipelineError("Invalid --corners. Expected exactly four x,y pairs")
    return order_corners(np.asarray(pts, np.float32))


def video_properties(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PipelineError(f"OpenCV could not open {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if count <= 0 or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise PipelineError(f"Invalid video metadata: frames={count}, fps={fps}, size={width}x{height}")
    return count, fps, width, height


def sample_frames(path: Path, frame_count: int, n: int = 31) -> tuple[np.ndarray, np.ndarray]:
    indices = np.unique(np.linspace(0, frame_count - 1, min(n, frame_count), dtype=int))
    frames = []
    cap = cv2.VideoCapture(str(path))
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if len(frames) < 3:
        raise PipelineError("Could not decode at least three calibration frames")
    return indices[: len(frames)], np.stack(frames)


def detect_arena(median_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = median_bgr.shape[:2]
    gray = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    candidates: list[tuple[float, np.ndarray, str]] = []
    for low, high in ((30, 90), (50, 150), (80, 220)):
        edges = cv2.Canny(gray, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            area = abs(cv2.contourArea(approx))
            ratio = area / float(w * h)
            if len(approx) == 4 and cv2.isContourConvex(approx) and 0.20 <= ratio <= 0.98:
                pts = order_corners(approx[:, 0, :])
                side_lengths = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
                if min(side_lengths) < min(h, w) * 0.15:
                    continue
                border_margin = max(3.0, 0.01 * min(w, h))
                border_penalty = sum(
                    min(x, y, w - 1 - x, h - 1 - y) < border_margin for x, y in pts
                ) * 0.12
                score = ratio - border_penalty
                candidates.append((score, pts, f"canny_{low}_{high}"))
    if not candidates:
        # Hough-informed fallback: bounding rotated rectangle of the dominant edge structure.
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 0.1 * w * h]
        if contours:
            rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
            pts = order_corners(cv2.boxPoints(rect))
            candidates.append((cv2.contourArea(max(contours, key=cv2.contourArea)) / (w * h), pts, "min_area_rect"))
    if not candidates:
        raise PipelineError(
            "Automatic arena detection found no credible quadrilateral. Re-run with "
            "--corners 'TLx,TLy;TRx,TRy;BRx,BRy;BLx,BLy'."
        )
    score, corners, method = max(candidates, key=lambda item: item[0])
    return corners, {"method": method, "candidate_count": len(candidates), "score": round(float(score), 6)}


def perspective_geometry(corners: np.ndarray, arena_w_cm: float, arena_h_cm: float):
    out_w = max(100, int(round(arena_w_cm * RECTIFIED_PX_PER_CM)))
    out_h = max(100, int(round(arena_h_cm * RECTIFIED_PX_PER_CM)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    inverse = cv2.getPerspectiveTransform(dst, corners)
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    cm_per_px_x = arena_w_cm / max((top + bottom) / 2.0, 1e-6)
    cm_per_px_y = arena_h_cm / max((left + right) / 2.0, 1e-6)
    return out_w, out_h, matrix, inverse, cm_per_px_x, cm_per_px_y


def robust_threshold(rectified_samples: np.ndarray, background: np.ndarray, requested: float) -> tuple[float, dict]:
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    values = []
    gray_samples = []
    raw_max = 0
    for frame in rectified_samples:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_samples.append(gray)
        delta = cv2.absdiff(gray, gray_bg)
        raw_max = max(raw_max, int(delta.max()))
        values.append(delta.reshape(-1))
    all_delta = np.concatenate(values).astype(np.float32)
    temporal_range = np.ptp(np.stack(gray_samples).astype(np.int16), axis=0)
    median = float(np.median(all_delta))
    mad = float(np.median(np.abs(all_delta - median)))
    estimated = float(np.clip(max(12.0, median + 8.0 * max(mad, 1.0)), 12.0, 60.0))
    threshold = float(requested if requested > 0 else estimated)
    stats = {
        "raw_max_abs_difference": raw_max,
        "median_abs_difference": round(median, 3),
        "p95_abs_difference": round(float(np.percentile(all_delta, 95)), 3),
        "p99_abs_difference": round(float(np.percentile(all_delta, 99)), 3),
        "temporal_range_p90": round(float(np.percentile(temporal_range, 90)), 3),
        "temporal_range_p99": round(float(np.percentile(temporal_range, 99)), 3),
        "estimated_threshold": round(estimated, 3),
        "used_threshold": round(threshold, 3),
    }
    return threshold, stats


@dataclass
class Detection:
    body: tuple[float, float] | None
    tips: tuple[np.ndarray, np.ndarray] | None
    contour: np.ndarray | None
    area: float


def segment_mouse(frame: np.ndarray, background: np.ndarray, threshold: float, previous: tuple[float, float] | None) -> Detection:
    diff = cv2.absdiff(frame, background)
    magnitude = np.max(diff, axis=2)
    mask = (magnitude >= threshold).astype(np.uint8) * 255
    margin = max(3, int(round(min(mask.shape) * 0.01)))
    mask[:margin] = mask[-margin:] = 0
    mask[:, :margin] = mask[:, -margin:] = 0
    k = max(3, int(round(min(mask.shape) * 0.012)) | 1)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 2 | 1), max(3, k // 2 | 1)))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large, iterations=2)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    arena_area = mask.size
    options = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 0.0004 * arena_area <= area <= 0.20 * arena_area:
            center = centroids[label]
            distance = 0.0 if previous is None else np.linalg.norm(center - np.asarray(previous))
            options.append((area - 2.0 * distance, label, area, center))
    if not options:
        return Detection(None, None, None, 0.0)
    _, label, area, center = max(options, key=lambda x: x[0])
    component = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    pts = contour[:, 0, :].astype(np.float32)
    centered = pts - center.astype(np.float32)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    projection = centered @ axis
    tip_a = pts[int(np.argmin(projection))]
    tip_b = pts[int(np.argmax(projection))]
    return Detection((float(center[0]), float(center[1])), (tip_a, tip_b), contour, float(area))


class HeadTracker:
    def __init__(self, fps: float):
        self.fps = fps
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2, np.float32)
        self.previous_body: np.ndarray | None = None
        self.orientation: np.ndarray | None = None
        self.missing_count = 0

    def update(self, tips: tuple[np.ndarray, np.ndarray] | None, body: tuple[float, float] | None) -> tuple[float, float] | None:
        if tips is None or body is None:
            self.missing_count += 1
            if self.position is not None:
                self.position = self.position + self.velocity
                self.velocity *= 0.8
                if self.missing_count <= max(1, int(round(0.5 * self.fps))):
                    return tuple(map(float, self.position))
            return None
        self.missing_count = 0
        candidates = [np.asarray(tips[0], np.float32), np.asarray(tips[1], np.float32)]
        body_arr = np.asarray(body, np.float32)
        body_velocity = np.zeros(2, np.float32) if self.previous_body is None else body_arr - self.previous_body
        body_speed = float(np.linalg.norm(body_velocity))
        self.previous_body = body_arr.copy()
        if self.position is None:
            # Deterministic initial choice; temporal motion resolves orientation on following frames.
            chosen = min(candidates, key=lambda p: (p[0], p[1]))
            self.position = chosen.copy()
            side = chosen - body_arr
            self.orientation = side / max(float(np.linalg.norm(side)), 1e-6)
            return tuple(map(float, self.position))
        predicted = self.position + self.velocity
        if body_speed >= 0.75:
            movement_direction = body_velocity / body_speed
            # During locomotion the snout is normally on the leading side. This
            # deliberately resolves an arbitrary head/tail choice made at rest.
            chosen = max(
                candidates,
                key=lambda p: float(np.dot((p - body_arr) / max(np.linalg.norm(p - body_arr), 1e-6), movement_direction)),
            )
        else:
            # At low speed, retain anatomical orientation instead of following
            # noisy centroid changes during grooming or rearing.
            def stationary_cost(point: np.ndarray) -> float:
                side = point - body_arr
                side /= max(float(np.linalg.norm(side)), 1e-6)
                orientation_penalty = 0.0 if self.orientation is None else 12.0 * (1.0 - float(np.dot(side, self.orientation)))
                return float(np.linalg.norm(point - predicted)) + orientation_penalty
            chosen = min(candidates, key=stationary_cost)
        chosen_side = chosen - body_arr
        chosen_side_unit = chosen_side / max(float(np.linalg.norm(chosen_side)), 1e-6)
        polarity_flip = self.orientation is not None and float(np.dot(chosen_side_unit, self.orientation)) < -0.25
        if polarity_flip and body_speed >= 0.75:
            # Re-initialize on the supported side rather than low-pass filtering
            # through the animal, which would temporarily mark the centroid.
            self.position = body_arr + 0.65 * chosen_side
            self.velocity *= 0.25
            self.orientation = chosen_side_unit
            return tuple(map(float, self.position))
        # Reject a polarity flip unless it is materially closer to the prediction.
        max_step = max(8.0, 2.5 * np.linalg.norm(self.velocity) + 4.0)
        delta = chosen - self.position
        if np.linalg.norm(delta) > max_step:
            delta *= max_step / max(np.linalg.norm(delta), 1e-6)
        updated = self.position + 0.45 * delta
        new_velocity = updated - self.position
        self.velocity = 0.65 * self.velocity + 0.35 * new_velocity
        self.position = updated
        # Keep the filtered point on the selected head side of the body.
        if np.dot(self.position - body_arr, chosen - body_arr) < 0:
            self.position = body_arr + 0.5 * (chosen - body_arr)
        filtered_side = self.position - body_arr
        filtered_side /= max(float(np.linalg.norm(filtered_side)), 1e-6)
        if self.orientation is None:
            self.orientation = filtered_side
        else:
            mixed = 0.8 * self.orientation + 0.2 * filtered_side
            self.orientation = mixed / max(float(np.linalg.norm(mixed)), 1e-6)
        return tuple(map(float, self.position))


def rectified_to_cm(point: tuple[float, float] | None, width: int, height: int, arena_w: float, arena_h: float):
    if point is None:
        return (np.nan, np.nan)
    x = float(np.clip(point[0], 0, width - 1)) * arena_w / (width - 1)
    y = float(np.clip(point[1], 0, height - 1)) * arena_h / (height - 1)
    return x, y


def transform_point(point: tuple[float, float], matrix: np.ndarray) -> tuple[int, int]:
    arr = np.asarray([[point]], np.float32)
    out = cv2.perspectiveTransform(arr, matrix)[0, 0]
    return int(round(out[0])), int(round(out[1]))


def fill_short_gaps(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    limit = max(1, int(round(fps * 0.5)))
    columns = ["body_x_cm", "body_y_cm", "head_x_cm", "head_y_cm"]
    df[columns] = df[columns].interpolate(limit=limit, limit_area="inside")
    return df


def write_annotated_video(
    input_path: Path, output_path: Path, df: pd.DataFrame, fps: float, size: tuple[int, int],
    corners: np.ndarray, inverse: np.ndarray, rect_size: tuple[int, int], background: np.ndarray,
    threshold: float, draw_contour: bool, max_frames: int, arena_width_cm: float, arena_height_cm: float,
) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise PipelineError("Could not initialize MP4 output writer")
    cap = cv2.VideoCapture(str(input_path))
    trail = np.zeros((height, width, 3), np.uint8)
    previous_draw = None
    previous_detect = None
    for i, row in df.iterrows():
        if max_frames and i >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        clean_frame = frame.copy()
        body_cm = (row.body_x_cm, row.body_y_cm)
        if np.all(np.isfinite(body_cm)):
            rx = body_cm[0] / arena_width_cm * (rect_size[0] - 1)
            ry = body_cm[1] / arena_height_cm * (rect_size[1] - 1)
            current_draw = transform_point((rx, ry), inverse)
            if previous_draw is not None:
                cv2.line(trail, previous_draw, current_draw, (0, 180, 255), 2, cv2.LINE_AA)
            previous_draw = current_draw
        cv2.addWeighted(frame, 1.0, trail, 0.75, 0, frame)
        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        rectified = cv2.warpPerspective(clean_frame, cv2.getPerspectiveTransform(corners, np.array([[0,0],[rect_size[0]-1,0],[rect_size[0]-1,rect_size[1]-1],[0,rect_size[1]-1]],np.float32)), rect_size)
        detection = segment_mouse(rectified, background, threshold, previous_detect)
        if detection.body is not None:
            previous_detect = detection.body
        if draw_contour and detection.contour is not None:
            contour_src = cv2.perspectiveTransform(detection.contour.astype(np.float32), inverse).astype(np.int32)
            cv2.drawContours(frame, [contour_src], -1, (255, 180, 0), 1, cv2.LINE_AA)
        head_cm = (row.head_x_cm, row.head_y_cm)
        if np.all(np.isfinite(head_cm)):
            rx = head_cm[0] / arena_width_cm * (rect_size[0] - 1)
            ry = head_cm[1] / arena_height_cm * (rect_size[1] - 1)
            cv2.circle(frame, transform_point((rx, ry), inverse), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, f"frame {int(row.frame)}  t={row.timestamp_sec:.2f}s", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / "pipeline_log.txt")
    logger.log("Mouse open-field tracking pipeline")
    logger.log("Status: STARTED")
    try:
        input_path = resolve_input(args.input, output_dir)
        frame_count, fps, frame_w, frame_h = video_properties(input_path)
        process_count = min(frame_count, args.max_frames) if args.max_frames else frame_count
        logger.log(f"Input: {input_path}")
        logger.log(f"Video: {frame_count} frames, {fps:.6f} FPS, {frame_w}x{frame_h}, processing {process_count}")
        indices, samples = sample_frames(input_path, frame_count)
        median_frame = np.median(samples, axis=0).astype(np.uint8)
        if args.corners:
            corners, arena_info = parse_corners(args.corners), {"method": "user_override", "candidate_count": 1, "score": 1.0}
        else:
            corners, arena_info = detect_arena(median_frame)
        out_w, out_h, matrix, inverse, cm_pp_x, cm_pp_y = perspective_geometry(
            corners, args.arena_width_cm, args.arena_height_cm
        )
        rectified_samples = np.stack([cv2.warpPerspective(f, matrix, (out_w, out_h)) for f in samples])
        background = np.median(rectified_samples, axis=0).astype(np.uint8)
        threshold, static_stats = robust_threshold(rectified_samples, background, args.threshold)
        logger.log(f"Arena: {arena_info['method']}, corners={corners.round(2).tolist()}")
        logger.log(f"Static-background diagnostics (mouse/compression included): {static_stats}")
        logger.log(f"Segmentation threshold: {threshold:.2f}")
        calibration = {
            "arena_corners_px": corners.round(3).tolist(),
            "cm_per_px_x": float(cm_pp_x),
            "cm_per_px_y": float(cm_pp_y),
            "arena_width_cm": float(args.arena_width_cm),
            "arena_height_cm": float(args.arena_height_cm),
            "perspective_transform": matrix.tolist(),
            "rectified_size_px": [out_w, out_h],
            "rectified_cm_per_px_x": float(args.arena_width_cm / (out_w - 1)),
            "rectified_cm_per_px_y": float(args.arena_height_cm / (out_h - 1)),
            "detection": arena_info,
            "sampled_frames": indices.astype(int).tolist(),
            "static_background_diagnostics": static_stats,
        }
        (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        cv2.imwrite(str(output_dir / "median_background.png"), background)
        preview = median_frame.copy()
        cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / "arena_calibration_preview.png"), preview)

        cap = cv2.VideoCapture(str(input_path))
        head_tracker = HeadTracker(fps)
        rows = []
        previous_body = None
        areas = []
        missing = 0
        for frame_index in range(process_count):
            ok, frame = cap.read()
            if not ok:
                logger.log(f"WARNING: decoder stopped at frame {frame_index}")
                break
            rectified = cv2.warpPerspective(frame, matrix, (out_w, out_h))
            detection = segment_mouse(rectified, background, threshold, previous_body)
            if detection.body is None:
                missing += 1
                head = head_tracker.update(None, None)
            else:
                previous_body = detection.body
                areas.append(detection.area)
                head = head_tracker.update(detection.tips, detection.body)
            bx, by = rectified_to_cm(detection.body, out_w, out_h, args.arena_width_cm, args.arena_height_cm)
            hx, hy = rectified_to_cm(head, out_w, out_h, args.arena_width_cm, args.arena_height_cm)
            rows.append((frame_index, frame_index / fps, bx, by, hx, hy))
            if frame_index and frame_index % 1000 == 0:
                logger.log(f"Tracking progress: {frame_index}/{process_count}")
        cap.release()
        df = pd.DataFrame(rows, columns=["frame", "timestamp_sec", "body_x_cm", "body_y_cm", "head_x_cm", "head_y_cm"])
        df = fill_short_gaps(df, fps)
        df.to_csv(output_dir / "trajectory.csv", index=False, float_format="%.6f")
        logger.log(f"Tracking complete: {len(df)} rows, raw missing body detections={missing} ({100*missing/max(len(df),1):.2f}%)")
        write_annotated_video(
            input_path, output_dir / "annotated_output.mp4", df, fps, (frame_w, frame_h), corners,
            inverse, (out_w, out_h), background, threshold, not args.no_contour, args.max_frames,
            args.arena_width_cm, args.arena_height_cm,
        )
        valid = df.dropna()
        if len(valid) > 1:
            body_step = np.hypot(np.diff(valid.body_x_cm), np.diff(valid.body_y_cm))
            head_body = np.hypot(valid.head_x_cm - valid.body_x_cm, valid.head_y_cm - valid.body_y_cm)
            logger.log(
                f"QA: body speed p99={np.percentile(body_step*fps,99):.2f} cm/s; "
                f"head-body distance median={np.median(head_body):.2f} cm, p99={np.percentile(head_body,99):.2f} cm"
            )
        if areas:
            logger.log(f"QA: mouse mask area median={np.median(areas):.1f} px, CV={np.std(areas)/max(np.mean(areas),1):.3f}")
        logger.log("Outputs: calibration.json, trajectory.csv, annotated_output.mp4")
        logger.log("Status: SUCCESS")
    except Exception as exc:
        logger.log(f"Status: FAILED - {exc}")
        logger.log(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        sys.exit(1)
