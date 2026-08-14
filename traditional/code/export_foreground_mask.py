#!/usr/bin/env python3
"""Export a binary foreground-mask video before body/head tracking.

White pixels are dark foreground relative to a high-percentile static
background: mouse, tail, miniscope, fibre, and any remaining dark moving object.
Black pixels are removed background. No torso extraction, cable removal, or
head estimation is applied in this diagnostic stage.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corners", default="", help="TLx,TLy;TRx,TRy;BRx,BRy;BLx,BLy")
    parser.add_argument("--roi-json", default="", help="Alternative saved four-corner ROI JSON")
    parser.add_argument("--arena-width-cm", type=float, required=True)
    parser.add_argument("--arena-height-cm", type=float, required=True)
    parser.add_argument("--background-percentile", type=float, default=85.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--sample-count", type=int, default=61)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Diagnostic duration; 0 exports full video")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if args.roi_json:
        corners = np.asarray(json.loads(Path(args.roi_json).read_text(encoding="utf-8"))["arena_corners_px"], np.float32)
    elif args.corners:
        corners = parse_corners(args.corners)
    else:
        raise ValueError("Provide --corners or --roi-json")
    width = max(100, round(args.arena_width_cm * 10))
    height = max(100, round(args.arena_height_cm * 10))
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(corners, destination)

    sample_indices = np.unique(np.linspace(0, nframes - 1, min(args.sample_count, nframes), dtype=int))
    samples = []
    for index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            samples.append(cv2.warpPerspective(frame, homography, (width, height)))
    if len(samples) < 3:
        raise RuntimeError("Could not decode enough frames for a background")
    percentile = float(np.clip(args.background_percentile, 50, 100))
    background = np.percentile(np.stack(samples), percentile, axis=0).astype(np.uint8)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    differences = []
    for frame in samples:
        differences.append(cv2.subtract(gray_background, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).reshape(-1))
    values = np.concatenate(differences).astype(np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = args.threshold if args.threshold > 0 else float(np.clip(max(12, median + 8 * max(mad, 1)), 12, 60))

    cv2.imwrite(str(output_dir / "foreground_background.png"), background)
    writer = cv2.VideoWriter(
        str(output_dir / "foreground_mask.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create foreground_mask.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    foreground_fraction = []
    max_frames = min(nframes, int(round(args.max_seconds * fps))) if args.max_seconds > 0 else nframes
    for index in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        rectified = cv2.warpPerspective(frame, homography, (width, height))
        dark_difference = cv2.subtract(background, rectified)
        strength = np.max(dark_difference, axis=2)
        mask = (strength >= threshold).astype(np.uint8) * 255
        # Only suppress the outermost rectification interpolation boundary. The
        # fibre and tail are intentionally retained without morphology.
        mask[:2] = mask[-2:] = 0
        mask[:, :2] = mask[:, -2:] = 0
        foreground_fraction.append(float(np.mean(mask > 0)))
        writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    cap.release()
    writer.release()
    metadata = {
        "input": str(input_path.resolve()),
        "frames_written": len(foreground_fraction),
        "fps": fps,
        "rectified_mask_size_px": [width, height],
        "arena_corners_px": corners.tolist(),
        "arena_width_cm": args.arena_width_cm,
        "arena_height_cm": args.arena_height_cm,
        "background_percentile": percentile,
        "foreground_threshold": threshold,
        "foreground_fraction_median": float(np.median(foreground_fraction)),
        "foreground_fraction_p99": float(np.percentile(foreground_fraction, 99)),
        "notes": "White = unfiltered dark foreground; black = removed background.",
    }
    (output_dir / "foreground_mask_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
