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
    title = f"Select activity-floor corners: {input_path.name}"
    points: list[tuple[int, int]] = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, click)
    while True:
        preview = frame.copy()
        instruction = "Click TL, TR, BR, BL clockwise. Right-click undo. Enter/Space confirm; C cancel."
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 31), (0, 0, 0), -1)
        cv2.putText(preview, instruction, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        labels = ("TL", "TR", "BR", "BL")
        for index, point in enumerate(points):
            cv2.circle(preview, point, 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(preview, labels[index], (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
        if len(points) > 1:
            cv2.polylines(preview, [np.asarray(points, np.int32)], len(points) == 4, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(title, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(points) == 4:
            break
        if key in (ord("c"), 27):
            cv2.destroyWindow(title)
            raise RuntimeError("ROI selection cancelled")
    cv2.destroyWindow(title)
    return [[float(x), float(y)] for x, y in points]


def select_initial_body(input_path: Path) -> tuple[float, float]:
    """One mouse-body click supplies identity when fibre is also foreground."""
    cap = cv2.VideoCapture(str(input_path)); ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError(f"Cannot read first frame: {input_path}")
    title = f"Click mouse body (not fibre): {input_path.name}"
    point: list[tuple[int, int]] = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: point[:] = [(x, y)]
    cv2.namedWindow(title, cv2.WINDOW_NORMAL); cv2.setMouseCallback(title, click)
    while True:
        preview = frame.copy()
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(preview, "Click compact mouse body/miniscope (not fibre); Enter confirm; C cancel", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
        if point: cv2.circle(preview, point[0], 7, (0,255,255), -1, cv2.LINE_AA)
        cv2.imshow(title, preview); key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and point: break
        if key in (27, ord('c')): cv2.destroyWindow(title); raise RuntimeError("Mouse-body seed cancelled")
    cv2.destroyWindow(title)
    return float(point[0][0]), float(point[0][1])


def select_body_at_time(input_path: Path, seconds: float) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(input_path)); fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps)); ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError(f"Cannot read reseed frame at {seconds:.2f}s")
    title = f"Click mouse after manual move ({seconds:.2f}s)"
    point: list[tuple[int, int]] = []
    def click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN: point[:] = [(x,y)]
    cv2.namedWindow(title,cv2.WINDOW_NORMAL); cv2.setMouseCallback(title,click)
    while True:
        preview=frame.copy(); cv2.rectangle(preview,(0,0),(preview.shape[1],30),(0,0,0),-1)
        cv2.putText(preview,"Click compact mouse body/miniscope after move; Enter confirm",(8,21),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
        if point: cv2.circle(preview,point[0],7,(0,255,255),-1)
        cv2.imshow(title,preview); key=cv2.waitKey(20)&255
        if key in (13,32) and point: break
        if key in (27,ord('c')): cv2.destroyWindow(title); raise RuntimeError('Reseed cancelled')
    cv2.destroyWindow(title); return float(point[0][0]),float(point[0][1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="", help="Previously trained .joblib head calibrator; not needed with --prepare-only")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after ROI/body seed and automatic candidates; use before labelling a first project")
    parser.add_argument("--output-dir", default="", help="Defaults to results/<video-name>_inference")
    parser.add_argument("--arena-width-cm", type=float, required=True)
    parser.add_argument("--arena-height-cm", type=float, required=True)
    parser.add_argument("--roi-json", default="", help="Reuse an existing ROI instead of selecting one")
    parser.add_argument("--initial-body-px", default="", help="Reuse mouse-body seed x,y; omit to click once")
    parser.add_argument("--reseed-sec", action="append", type=float, default=[], help="Time after a manual mouse move; click a new identity anchor (repeatable)")
    parser.add_argument("--interactive-recovery", action="store_true", help="Pause only after sustained body-tracking loss and request confirmation/reseed")
    parser.add_argument("--recovery-gap-sec", type=float, default=.5)
    parser.add_argument("--background-percentile", type=float, default=100, help="Use 100 for dark mouse/fibre on a bright floor; avoids mouse-contaminated backgrounds")
    parser.add_argument("--max-gap-sec", type=float, default=.5)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    if not source.is_file(): raise FileNotFoundError(source)
    model = Path(args.model).resolve() if args.model else None
    if not args.prepare_only and (model is None or not model.is_file()):
        raise FileNotFoundError("Provide an existing --model, or use --prepare-only before first training")
    out = Path(args.output_dir).resolve() if args.output_dir else Path("results") / f"{source.stem}_inference"
    out.mkdir(parents=True, exist_ok=True)
    roi = Path(args.roi_json).resolve() if args.roi_json else out / "arena_roi.json"
    if not args.roi_json:
        corners = select_roi(source)
        roi.write_text(json.dumps({"input": str(source), "arena_corners_px": corners, "selection_mode": "manual_four_corner_perspective"}, indent=2), encoding="utf-8")
    if not roi.is_file(): raise FileNotFoundError(roi)
    if args.initial_body_px:
        initial_body = args.initial_body_px
    else:
        x, y = select_initial_body(source)
        initial_body = f"{x},{y}"
    reseed_args=[]
    fps=cv2.VideoCapture(str(source)).get(cv2.CAP_PROP_FPS)
    for seconds in args.reseed_sec:
        x,y=select_body_at_time(source,seconds)
        reseed_args.extend(["--reseed", f"{round(seconds*fps)}:{x},{y}"])
    script_dir = Path(__file__).parent
    def run(script: str, *values: str) -> None:
        subprocess.run([sys.executable, str(script_dir / script), *values], check=True)
    common = ["--input", str(source), "--output-dir", str(out), "--roi-json", str(roi), "--arena-width-cm", str(args.arena_width_cm), "--arena-height-cm", str(args.arena_height_cm)]
    recovery_args = ["--interactive-recovery", "--recovery-gap-sec", str(args.recovery_gap_sec)] if args.interactive_recovery else []
    run("compare_head_methods.py", *common, "--background-percentile", str(args.background_percentile), "--repair-background", "--initial-body-px", initial_body, *reseed_args, *recovery_args)
    if args.prepare_only:
        print(f"\nPREPARATION COMPLETE\nCandidates: {out / 'head_method_comparison.csv'}\nROI: {roi}")
        return
    run("finalize_head_trajectory.py", "--comparison-csv", str(out / "head_method_comparison.csv"), "--model", str(model), "--output", str(out / "trajectory_inference.csv"), "--input", str(source), "--roi-json", str(roi), "--arena-width-cm", str(args.arena_width_cm), "--arena-height-cm", str(args.arena_height_cm), "--max-gap-sec", str(args.max_gap_sec), "--output-video", str(out / "annotated_inference.mp4"))
    print(f"\nCOMPLETE\nCSV: {out / 'trajectory_inference.csv'}\nVideo: {out / 'annotated_inference.mp4'}\nROI: {roi}")


if __name__ == "__main__": main()
