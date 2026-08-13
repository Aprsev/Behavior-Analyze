#!/usr/bin/env python3
"""Compare silhouette and miniscope-reflection head anchors on one video.

The reflection is detected from the *raw* image, rather than the dark
background-subtraction mask.  A candidate must be close to the compact mouse
torso and temporally continuous, so static highlights and wall markings are
discarded.  This is an analysis/QA tool: it deliberately produces both tracks
side-by-side instead of silently replacing the existing head estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mouse_behavior_pipeline import (
    HeadTracker, parse_corners, perspective_geometry, robust_threshold,
    sample_frames, segment_mouse, transform_point, video_properties,
)


def roi_corners(args: argparse.Namespace) -> np.ndarray:
    if args.corners:
        return parse_corners(args.corners)
    if args.roi_json:
        data = json.loads(Path(args.roi_json).read_text(encoding="utf-8"))
        return parse_corners(";".join(f"{x},{y}" for x, y in data["arena_corners_px"]))
    raise ValueError("Provide --corners or --roi-json")


class ReflectionTracker:
    """Track the brightest compact spot mounted on/very near the mouse body."""

    def __init__(self, fps: float):
        self.fps = fps
        self.position: np.ndarray | None = None
        self.relative: np.ndarray | None = None
        self.missing = 0

    def update(self, frame: np.ndarray, background: np.ndarray, body: tuple[float, float] | None,
               body_mask: np.ndarray | None) -> tuple[tuple[float, float] | None, float, str]:
        if body is None or body_mask is None:
            self.missing += 1
            if self.missing > self.fps * 0.5:
                self.position = self.relative = None
            return None, 0.0, "no_body"
        self.missing = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        # Specular reflection: it is both bright in absolute intensity and
        # noticeably brighter than the temporal background at that location.
        bright_delta = cv2.subtract(gray, bg_gray)
        # Top-hat retains small specular spots even when the absolute reflection
        # is modest or the whole mouse/miniscope region is bright.
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        local = cv2.dilate(body_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
        pixels = bright_delta[local > 0]
        if pixels.size == 0:
            return None, 0.0, "no_search_region"
        # Weak mode: lower brightness threshold but demand local bright-spot
        # evidence. This recovers dim reflections while the body-relative gate
        # below protects against unrelated arena features.
        delta_threshold = max(4.0, float(np.percentile(pixels, 74)))
        absolute_threshold = max(95.0, float(np.percentile(gray[local > 0], 55)))
        top_hat_threshold = max(5.0, float(np.percentile(top_hat[local > 0], 70)))
        candidate = (
            (gray >= absolute_threshold)
            & (local > 0)
            & ((bright_delta >= delta_threshold) | (top_hat >= top_hat_threshold))
        ).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, labels, stats, centers = cv2.connectedComponentsWithStats(candidate)
        body_arr = np.asarray(body, np.float32)
        choices = []
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not 1 <= area <= 260:
                continue
            center = centers[label].astype(np.float32)
            dist_body = float(np.linalg.norm(center - body_arr))
            if dist_body > 52:
                continue
            intensity = float(np.mean(bright_delta[labels == label]))
            predicted = body_arr + self.relative if self.relative is not None else body_arr
            continuity = float(np.linalg.norm(center - predicted))
            if self.position is not None and continuity > 28:
                continue
            # High contrast is evidence for a reflection; closeness to its
            # previous body-relative attachment point prevents spot switching.
            local_strength = float(np.mean(top_hat[labels == label]))
            score = 2.0 * intensity + 1.2 * local_strength - 1.0 * continuity - 0.15 * dist_body
            choices.append((score, center, intensity, area))
        if not choices:
            return None, 0.0, "reflection_missing"
        _, observed, intensity, area = max(choices, key=lambda item: item[0])
        if self.position is None:
            filtered = observed
        else:
            filtered = 0.65 * self.position + 0.35 * observed
        self.position = filtered.astype(np.float32)
        relative_observed = observed - body_arr
        self.relative = relative_observed if self.relative is None else 0.9 * self.relative + 0.1 * relative_observed
        confidence = float(np.clip((intensity - 3) / 35, 0, 1) * np.clip(area / 3, 0.3, 1))
        return tuple(map(float, self.position)), confidence, "reflection"


def to_cm(point, width, height, arena_w, arena_h):
    if point is None:
        return np.nan, np.nan
    return point[0] * arena_w / (width - 1), point[1] * arena_h / (height - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-json", default="")
    parser.add_argument("--corners", default="")
    parser.add_argument("--arena-width-cm", type=float, required=True)
    parser.add_argument("--arena-height-cm", type=float, required=True)
    parser.add_argument("--background-percentile", type=float, default=85)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--torso-labels", default="", help="Optional manual torso-box CSV; applies exact fibre exclusion on labelled frames")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count, fps, source_w, source_h = video_properties(source)
    count = min(count, args.max_frames) if args.max_frames else count
    corners = roi_corners(args)
    out_w, out_h, forward, inverse, _, _ = perspective_geometry(corners, args.arena_width_cm, args.arena_height_cm)
    _, samples = sample_frames(source, count, 61)
    rect_samples = np.stack([cv2.warpPerspective(f, forward, (out_w, out_h)) for f in samples])
    background = np.percentile(rect_samples, np.clip(args.background_percentile, 50, 100), axis=0).astype(np.uint8)
    threshold, _ = robust_threshold(rect_samples, background, args.threshold)
    torso_labels = pd.DataFrame()
    if args.torso_labels:
        torso_labels = pd.read_csv(args.torso_labels)
        # Polygon labels replace the earlier rectangle CSV format. Keep a
        # backwards-compatible rectangle reader for any existing annotations.
        torso_labels = torso_labels.loc[~torso_labels.exclude.astype(bool)].set_index("frame")
    cv2.imwrite(str(output / "reflection_background.png"), background)

    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(output / "head_method_comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (source_w, source_h))
    silhouette = HeadTracker(fps)
    reflection = ReflectionTracker(fps)
    previous_body = previous_bbox = None
    missing = 0
    rows = []
    max_jump = max(8.0, 20 * 30.0 / fps)
    for index in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        rectified = cv2.warpPerspective(frame, forward, (out_w, out_h))
        allowed_mask = None
        if index in torso_labels.index:
            label = torso_labels.loc[index]
            if "polygon_px" in torso_labels.columns and isinstance(label.polygon_px, str) and label.polygon_px:
                polygon = np.asarray(json.loads(label.polygon_px), np.float32)
                rect_box = cv2.perspectiveTransform(polygon[None], forward)[0].astype(np.int32)
                allowed_mask = np.zeros((out_h, out_w), np.uint8)
                cv2.fillPoly(allowed_mask, [rect_box], 255)
            elif {"x", "y", "width", "height"}.issubset(torso_labels.columns) and np.isfinite([label.x, label.y, label.width, label.height]).all():
                source_box = np.asarray([[[label.x, label.y], [label.x + label.width, label.y], [label.x + label.width, label.y + label.height], [label.x, label.y + label.height]]], np.float32)
                rect_box = cv2.perspectiveTransform(source_box, forward)[0].astype(np.int32)
                allowed_mask = np.zeros((out_h, out_w), np.uint8)
                cv2.fillConvexPoly(allowed_mask, rect_box, 255)
        detection = segment_mouse(rectified, background, threshold, previous_body, max_jump, previous_bbox, allowed_mask)
        if detection.body is None:
            missing += 1
            old = silhouette.update(None, None)
            new, confidence, source_name = reflection.update(rectified, background, None, None)
            if missing > fps * 0.5:
                previous_body = previous_bbox = None
                silhouette.reset()
        else:
            missing = 0
            previous_body, previous_bbox = detection.body, detection.bbox
            old = silhouette.update(detection.tips, detection.body, detection.head_hint_confidence)
            body_mask = np.zeros((out_h, out_w), np.uint8)
            cv2.drawContours(body_mask, [detection.contour], -1, 255, -1)
            new, confidence, source_name = reflection.update(rectified, background, detection.body, body_mask)
        bx, by = to_cm(detection.body, out_w, out_h, args.arena_width_cm, args.arena_height_cm)
        ox, oy = to_cm(old, out_w, out_h, args.arena_width_cm, args.arena_height_cm)
        rx, ry = to_cm(new, out_w, out_h, args.arena_width_cm, args.arena_height_cm)
        rows.append((index, index / fps, bx, by, ox, oy, rx, ry, confidence, source_name))
        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        if detection.contour is not None:
            cv2.drawContours(frame, [cv2.perspectiveTransform(detection.contour.astype(np.float32), inverse).astype(np.int32)], -1, (255, 180, 0), 1)
        for point, color, label in ((old, (0, 0, 255), "silhouette"), (new, (255, 0, 255), "reflection")):
            if point is not None:
                cv2.circle(frame, transform_point(point, inverse), 5, color, -1)
                cv2.putText(frame, label, tuple(np.asarray(transform_point(point, inverse)) + (7, -7)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
        cv2.putText(frame, f"{index}  reflect={source_name} ({confidence:.2f})", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2)
        writer.write(frame)
    cap.release(); writer.release()
    df = pd.DataFrame(rows, columns=["frame", "timestamp_sec", "body_x_cm", "body_y_cm", "head_silhouette_x_cm", "head_silhouette_y_cm", "head_reflection_x_cm", "head_reflection_y_cm", "reflection_confidence", "reflection_status"])
    df.to_csv(output / "head_method_comparison.csv", index=False, float_format="%.5f")
    meta = {"input": str(source.resolve()), "frames_processed": len(df), "fps": fps, "arena_corners_px": corners.tolist(), "threshold": threshold, "reflection_valid_frames": int(df.head_reflection_x_cm.notna().sum()), "reflection_valid_percent": round(100 * float(df.head_reflection_x_cm.notna().mean()), 2)}
    (output / "comparison_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
