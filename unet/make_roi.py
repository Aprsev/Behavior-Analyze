#!/usr/bin/env python3
"""Click the four arena corners to create a ROI JSON for a new video.

Every downstream tool (compare / screen / annotate / infer / head) needs a
ROI JSON holding the arena corners in source-frame pixels. For a new
recording, run this once:

    python unet/run_unet.py roi --video "data/new.avi"

or directly:

    python unet/make_roi.py --video "data/new.avi"

Controls: left-click adds a corner (4 needed, any order); Backspace undoes
the last corner; R resets all; S saves the ROI JSON; Q/Esc quits without
saving. Saved format matches traditional/basic_rois/*.json.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import refine_corners  # noqa: E402


def default_output(video: Path) -> Path:
    """traditional/basic_rois/{stem}_roi.json relative to the repo root."""
    root = Path(__file__).resolve().parents[1]
    return root / "traditional" / "basic_rois" / f"{video.stem}_roi.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="source video (.avi/.mp4)")
    p.add_argument("--output", default="", help="output JSON path (default: traditional/basic_rois/{stem}_roi.json)")
    p.add_argument("--min-dist", type=float, default=8.0, help="minimum px between clicks")
    a = p.parse_args()

    video = Path(a.video)
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read first frame of {video}")
    out = Path(a.output) if a.output else default_output(video)
    title = "Arena corners: click 4 corners | A auto-snap | Backspace undo | R reset | S save | Q quit"
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(pts) >= 4:
            return
        if all(np.hypot(x - px, y - py) > a.min_dist for px, py in pts):
            pts.append((int(x), int(y)))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed without saving.")
            return
        img = frame.copy()
        if len(pts) >= 2:
            poly = np.asarray(pts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"corners: {len(pts)}/4   (S to save to {out.name})",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(title, img)
        k = cv2.waitKey(20) & 255
        if k in (ord("q"), 27):
            print("Quit without saving.")
            return
        if k in (8, 127) and pts:
            pts.pop()
        if k == ord("r"):
            pts.clear()
        if k == ord("a") and len(pts) == 4:
            # Auto-calibration: snap the 4 clicks to the detected arena edges.
            refined = refine_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                     np.asarray(pts, np.float32))
            moved = sum(np.hypot(refined[i][0] - pts[i][0], refined[i][1] - pts[i][1]) > 1.5
                        for i in range(4))
            pts = [(int(round(x)), int(round(y))) for x, y in refined]
            print(f"A: snapped {moved}/4 corners to the detected arena edges")
        if k == ord("s"):
            if len(pts) != 4:
                print(f"Need exactly 4 corners; only {len(pts)} clicked.")
                continue
            payload = {
                "input": str(video),
                "arena_corners_px": [[float(x), float(y)] for x, y in pts],
                "selection_mode": "manual_four_corner_perspective",
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Saved ROI JSON: {out}")
            print("Next: python unet/run_unet.py compare/screen/... (edit VIDEOS in run_unet.py, or use --interactive)")
            return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()