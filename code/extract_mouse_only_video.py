#!/usr/bin/env python3
"""Extract a mouse-only video in the original camera field of view.

The output keeps the input resolution and timing. Background is black; only the
largest compact torso component is copied from each source frame. Thin attached
structures (tail and fibre) are removed by an adaptive morphological opening.
The head-mounted microscope is retained when it is part of the compact torso.
No head position is estimated in this extraction stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_corners(value: str) -> np.ndarray:
    points = [[float(x) for x in pair.split(",")] for pair in value.split(";")]
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("--corners must be TLx,TLy;TRx,TRy;BRx,BRy;BLx,BLy")
    return np.asarray(points, np.float32)


def select_arena_roi(frame: np.ndarray, video_name: str) -> np.ndarray:
    """Ask the operator to draw the activity-floor rectangle on the first frame."""
    title = f"Select white activity floor: {video_name}"
    display = frame.copy()
    cv2.putText(display, "Drag central white activity floor; Enter/Space = confirm, C = cancel", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    x, y, width, height = cv2.selectROI(title, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if width < 10 or height < 10:
        raise RuntimeError("Arena selection cancelled or too small")
    return np.asarray([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], np.float32)


def torso_mask(
    rectified: np.ndarray, background: np.ndarray, threshold: float,
    previous_center: np.ndarray | None, previous_bbox: tuple[int, int, int, int] | None,
    previous_area: float | None,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int, int, int] | None, float | None]:
    # The mouse/tether system is dark on a bright floor: only retain dark pixels.
    foreground = (np.max(cv2.subtract(background, rectified), axis=2) >= threshold).astype(np.uint8) * 255
    foreground[:2] = foreground[-2:] = 0
    foreground[:, :2] = foreground[:, -2:] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    arena_area = foreground.size
    candidates = []
    for label in range(1, count):
        full_area = int(stats[label, cv2.CC_STAT_AREA])
        if not (0.003 * arena_area <= full_area <= 0.18 * arena_area):
            continue
        full = (labels == label).astype(np.uint8) * 255
        radius = math.sqrt(full_area / math.pi)
        kernel_size = int(np.clip(round(0.35 * radius), 3, 15)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(full, cv2.MORPH_OPEN, kernel)
        n2, labels2, stats2, centers2 = cv2.connectedComponentsWithStats(opened)
        if n2 <= 1:
            continue
        selected = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        torso = (labels2 == selected).astype(np.uint8) * 255
        # Smooth the compact torso only; do not reattach fibre/tail.
        torso = cv2.morphologyEx(torso, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        torso = cv2.GaussianBlur(torso, (5, 5), 0)
        _, torso = cv2.threshold(torso, 127, 255, cv2.THRESH_BINARY)
        n3, labels3, stats3, centers3 = cv2.connectedComponentsWithStats(torso)
        if n3 <= 1:
            continue
        selected = 1 + int(np.argmax(stats3[1:, cv2.CC_STAT_AREA]))
        torso = (labels3 == selected).astype(np.uint8) * 255
        x, y, w, h, area = stats3[selected]
        center = centers3[selected]
        contours, _ = cv2.findContours(torso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = contour_area / max(hull_area, 1.0)
        aspect_ratio = max(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))
        ellipse_fill = 0.0
        if len(contour) >= 5:
            (_, _), axes, _ = cv2.fitEllipse(contour)
            ellipse_area = math.pi * axes[0] * axes[1] / 4.0
            ellipse_fill = contour_area / max(ellipse_area, 1.0)
        # A torso with a head-mounted device can be elongated, but a thin fibre
        # fragment is much more elongated and has weak compactness/solidity.
        if not (0.15 <= compactness <= 0.95 and aspect_ratio <= 4.5 and solidity >= 0.58):
            continue
        distance = 0.0 if previous_center is None else float(np.linalg.norm(center - previous_center))
        if previous_center is not None and distance > 12.0:
            continue
        compactness = float(area) / max(int(w) * int(h), 1)
        overlap = 0.0
        if previous_bbox is not None:
            px, py, pw, ph = previous_bbox
            margin_x, margin_y = max(12, pw), max(12, ph)
            ix0, iy0 = max(int(x), px - margin_x), max(int(y), py - margin_y)
            ix1, iy1 = min(int(x + w), px + pw + margin_x), min(int(y + h), py + ph + margin_y)
            overlap = max(0, ix1 - ix0) * max(0, iy1 - iy0) / max(int(w) * int(h), 1)
        area_change = 0.0 if previous_area is None else abs(math.log(max(float(area), 1.0) / max(previous_area, 1.0)))
        if previous_area is not None and area_change > math.log(2.8):
            continue
        shape_score = 2.0 * compactness + 1.5 * solidity + 0.5 * min(ellipse_fill, 1.0)
        score = float(area) + shape_score if previous_center is None else (
            4 * overlap + shape_score - distance / 12.0 - 0.8 * area_change
        )
        candidates.append((score, torso, center, (int(x), int(y), int(w), int(h)), float(area)))
    if not candidates:
        return None, None, None, None
    _, torso, center, bbox, area = max(candidates, key=lambda item: item[0])
    return torso, center, bbox, area


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="", help="Defaults to results/<input-video-name>")
    parser.add_argument("--corners", default="", help="TLx,TLy;TRx,TRy;BRx,BRy;BLx,BLy")
    parser.add_argument("--select-roi", action="store_true", help="Open a window to manually draw the activity-floor ROI")
    parser.add_argument("--arena-width-cm", type=float, required=True)
    parser.add_argument("--arena-height-cm", type=float, required=True)
    parser.add_argument("--background-percentile", type=float, default=85.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--sample-count", type=int, default=61)
    args = parser.parse_args()

    source = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / source.stem
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first_frame = capture.read()
    if not ok:
        raise RuntimeError("Cannot decode the first frame")
    if args.select_roi:
        corners = select_arena_roi(first_frame, source.name)
    elif args.corners:
        corners = parse_corners(args.corners)
    else:
        raise RuntimeError("Provide --select-roi or --corners")
    output_dir.mkdir(parents=True, exist_ok=True)
    rectified_width, rectified_height = round(args.arena_width_cm * 10), round(args.arena_height_cm * 10)
    destination = np.array([[0, 0], [rectified_width - 1, 0], [rectified_width - 1, rectified_height - 1], [0, rectified_height - 1]], np.float32)
    forward = cv2.getPerspectiveTransform(corners, destination)
    inverse = cv2.getPerspectiveTransform(destination, corners)

    sample_indices = np.unique(np.linspace(0, frame_count - 1, min(args.sample_count, frame_count), dtype=int))
    samples = []
    for index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            samples.append(cv2.warpPerspective(frame, forward, (rectified_width, rectified_height)))
    if len(samples) < 3:
        raise RuntimeError("Could not decode enough samples for the background")
    percentile = float(np.clip(args.background_percentile, 50, 100))
    background = np.percentile(np.stack(samples), percentile, axis=0).astype(np.uint8)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    differences = [
        cv2.subtract(gray_background, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).reshape(-1)
        for frame in samples
    ]
    values = np.concatenate(differences).astype(np.float32)
    median, mad = float(np.median(values)), float(np.median(np.abs(values - np.median(values))))
    threshold = args.threshold if args.threshold > 0 else float(np.clip(max(12, median + 8 * max(mad, 1)), 12, 60))
    cv2.imwrite(str(output_dir / "extraction_background.png"), background)

    output_path = output_dir / "mouse_only_original_fov.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (source_width, source_height))
    if not writer.isOpened():
        raise RuntimeError("Could not create output MP4")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    previous_center, previous_bbox, previous_area, missing = None, None, None, 0
    detected = 0
    for index in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        rectified = cv2.warpPerspective(frame, forward, (rectified_width, rectified_height))
        torso, center, bbox, area = torso_mask(
            rectified, background, threshold, previous_center, previous_bbox, previous_area
        )
        output = np.zeros_like(frame)
        if torso is not None:
            source_mask = cv2.warpPerspective(torso, inverse, (source_width, source_height), flags=cv2.INTER_NEAREST)
            output[source_mask > 0] = frame[source_mask > 0]
            previous_center, previous_bbox, previous_area, missing = center, bbox, area, 0
            detected += 1
        else:
            missing += 1
            if missing > round(0.5 * fps):
                previous_center, previous_bbox, previous_area = None, None, None
        writer.write(output)
    capture.release()
    writer.release()
    metadata = {
        "input": str(source.resolve()), "frames_written": frame_count, "fps": fps,
        "source_size_px": [source_width, source_height], "rectified_size_px": [rectified_width, rectified_height],
        "arena_corners_px": corners.tolist(), "background_percentile": percentile,
        "foreground_threshold": threshold, "detected_frames": detected,
        "missing_frames": frame_count - detected,
        "description": "Original-FOV video with black background and compact mouse torso only; tail/fibre removed.",
    }
    (output_dir / "mouse_only_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "arena_roi.json").write_text(json.dumps({
        "input": str(source.resolve()), "arena_corners_px": corners.tolist(),
        "selection_mode": "manual_rectangle" if args.select_roi else "command_line_corners",
    }, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
