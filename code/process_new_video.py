#!/usr/bin/env python3
"""One-command inference for a new video using an existing head calibrator.

It asks the operator to select the central activity-floor ROI once, then runs
automatic body/reflection candidates, the learned head model, gap completion,
and final annotated-video rendering. No manual head labels are needed for the
new video.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def select_roi(input_path: Path) -> list[list[float]]:
    cap = cv2.VideoCapture(str(input_path))
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {input_path}")
    title = f"Select central activity floor: {input_path.name}"
    cv2.putText(frame, "Drag the pure-white activity floor; Enter/Space confirm; C cancel", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, cv2.LINE_AA)
    x, y, width, height = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if width < 10 or height < 10:
        raise RuntimeError("ROI selection cancelled or too small")
    return [[float(x), float(y)], [float(x + width), float(y)], [float(x + width), float(y + height)], [float(x), float(y + height)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True, help="Previously trained .joblib head calibrator")
    parser.add_argument("--output-dir", default="", help="Defaults to results/<video-name>_inference")
    parser.add_argument("--arena-width-cm", type=float, required=True)
    parser.add_argument("--arena-height-cm", type=float, required=True)
    parser.add_argument("--roi-json", default="", help="Reuse an existing ROI instead of selecting one")
    parser.add_argument("--background-percentile", type=float, default=85)
    parser.add_argument("--max-gap-sec", type=float, default=.5)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    if not source.is_file(): raise FileNotFoundError(source)
    model = Path(args.model).resolve()
    if not model.is_file(): raise FileNotFoundError(model)
    out = Path(args.output_dir).resolve() if args.output_dir else Path("results") / f"{source.stem}_inference"
    out.mkdir(parents=True, exist_ok=True)
    roi = Path(args.roi_json).resolve() if args.roi_json else out / "arena_roi.json"
    if not args.roi_json:
        corners = select_roi(source)
        roi.write_text(json.dumps({"input": str(source), "arena_corners_px": corners, "selection_mode": "manual_rectangle_for_inference"}, indent=2), encoding="utf-8")
    if not roi.is_file(): raise FileNotFoundError(roi)
    script_dir = Path(__file__).parent
    def run(script: str, *values: str) -> None:
        subprocess.run([sys.executable, str(script_dir / script), *values], check=True)
    common = ["--input", str(source), "--output-dir", str(out), "--roi-json", str(roi), "--arena-width-cm", str(args.arena_width_cm), "--arena-height-cm", str(args.arena_height_cm)]
    run("compare_head_methods.py", *common, "--background-percentile", str(args.background_percentile))
    run("finalize_head_trajectory.py", "--comparison-csv", str(out / "head_method_comparison.csv"), "--model", str(model), "--output", str(out / "trajectory_inference.csv"), "--input", str(source), "--roi-json", str(roi), "--arena-width-cm", str(args.arena_width_cm), "--arena-height-cm", str(args.arena_height_cm), "--max-gap-sec", str(args.max_gap_sec), "--output-video", str(out / "annotated_inference.mp4"))
    print(f"\nCOMPLETE\nCSV: {out / 'trajectory_inference.csv'}\nVideo: {out / 'annotated_inference.mp4'}\nROI: {roi}")


if __name__ == "__main__": main()
