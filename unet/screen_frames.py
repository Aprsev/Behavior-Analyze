#!/usr/bin/env python3
"""Manual frame screening before annotation: pick diverse frames, flag junk.

For every input video this tool:
  1. builds the same background/threshold as the annotator;
  2. scans a uniform grid and splits frames into "good" (plausible mouse
     segmentation) and "junk" (segmentation failed, mouse absent, human
     intervention, motion blur, ...);
  3. farthest-point samples --per-video maximally different good frames;
  4. appends up to --junk junk frames, pre-excluded;
  5. shows a 3x3 montage: click a cell to toggle EXCLUDED (red border);
     junk frames start excluded and can be re-included if wrongly flagged.

Output CSV (one row per candidate; append across videos):
    video, frame, exclude

The same CSV is consumed by the annotator (--candidate-csv), by
prepare_dataset.py (--exclude-csv) and by infer.py (--exclude-csv), so the
screening decision is honoured in labelling, training and inference.

Keys: click cell toggle exclude | n/PageDown/space next page | p/PageUp prev
      | s save (keep browsing) | q/Esc save and quit
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "traditional" / "code"
sys.path.insert(0, str(CODE))
from annotate_torso_constraints import farthest_pick, scan_candidates  # noqa: E402
from mouse_behavior_pipeline import (  # noqa: E402
    perspective_geometry, robust_threshold, sample_frames, video_properties,
)

CELL_W, CELL_H, PAD = 426, 318, 4
GRID = 3
PAGE = GRID * GRID
CANVAS = (GRID * CELL_W + (GRID + 1) * PAD, GRID * CELL_H + (GRID + 1) * PAD + 30)


def load_corners(roi_json: Path) -> np.ndarray:
    return np.asarray(json.loads(Path(roi_json).read_text(encoding="utf-8"))["arena_corners_px"], np.float32)


def montage_session(cap: cv2.VideoCapture, items: list[dict], video_name: str) -> list[dict]:
    """Show 3x3 montage pages; user toggles excluded per candidate."""
    title = "Screen frames | click toggle EXCLUDE | n/p page | s save | q or X to finish"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL); cv2.resizeWindow(title, CANVAS)
    page = 0; total_pages = max(1, (len(items) + PAGE - 1) // PAGE)
    excluded = {i for i, it in enumerate(items) if it["junk"]}

    def mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        col = x // (CELL_W + PAD); row = (y - 30) // (CELL_H + PAD)
        idx = page * PAGE + row * GRID + col
        if 0 <= row < GRID and 0 <= col < GRID and idx < len(items):
            excluded.symmetric_difference_update({idx})

    cv2.setMouseCallback(title, mouse)
    while True:
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            # Window closed via the title-bar X: treat as save-and-quit.
            print("Window closed; screening results for this video are saved.")
            break
        canvas = np.full((CANVAS[1], CANVAS[0], 3), 240, np.uint8)
        cv2.putText(canvas, f"{video_name}  page {page+1}/{total_pages}  ({len(items)} candidates)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 40, 60), 1)
        for k in range(PAGE):
            idx = page * PAGE + k
            if idx >= len(items):
                break
            item = items[idx]; cap.set(cv2.CAP_PROP_POS_FRAMES, int(item["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            x0 = PAD + (k % GRID) * (CELL_W + PAD); y0 = 30 + PAD + (k // GRID) * (CELL_H + PAD)
            scale = min((CELL_W - 2 * PAD) / frame.shape[1], (CELL_H - 2 * PAD) / frame.shape[0])
            small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            ox = x0 + (CELL_W - small.shape[1]) // 2; oy = y0 + (CELL_H - small.shape[0]) // 2
            canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
            if item["contour"] is not None and not item["junk"]:
                pts = item["contour"].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, (0, 220, 0), 2, cv2.LINE_AA)
            label = f'f={item["frame"]}'
            if idx in excluded:
                cv2.rectangle(canvas, (x0, y0), (x0 + CELL_W, y0 + CELL_H), (0, 0, 220), 4)
                label += "  EXCLUDED"
            if item["junk"]:
                label += "  [junk]"
            cv2.putText(canvas, label, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 220) if idx in excluded else (20, 40, 60), 1)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(25) & 255
        if key in (ord("q"), 27): break
        if key in (ord("n"), ord(" ")): page = min(page + 1, total_pages - 1)
        if key in (ord("p"), ord("b")): page = max(page - 1, 0)
        if key == ord("s"):
            cv2.destroyWindow(title)
            return excluded
        if key == ord("e"):
            for k in range(PAGE):
                idx = page * PAGE + k
                if idx < len(items): excluded.symmetric_difference_update({idx})
    cv2.destroyAllWindows()
    return excluded


def save_rows(output: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=["video", "frame", "exclude"])
    if output.exists():
        old = pd.read_csv(output)
        df = pd.concat([old, df], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} screening rows to {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", action="append", required=True, help="source video (repeatable)")
    p.add_argument("--roi", action="append", required=True, help="ROI JSON (repeatable, one per video)")
    p.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "traditional" / "results" / "screening.csv"))
    p.add_argument("--per-video", type=int, default=40, help="diverse candidates to label per video")
    p.add_argument("--junk", type=int, default=20, help="junk frames shown per video (pre-excluded)")
    p.add_argument("--arena-width-cm", type=float, default=25.0)
    p.add_argument("--arena-height-cm", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true", help="print the selection without opening a window")
    a = p.parse_args()
    if len(a.video) != len(a.roi):
        p.error("--video and --roi must appear the same number of times")

    rows: list[dict] = []
    for video, roi in zip(a.video, a.roi):
        vp, rp = Path(video), Path(roi)
        print(f"[1/3] Background/threshold for {vp.name} ...", flush=True)
        corners = load_corners(rp)
        total, fps, _, _ = video_properties(vp)
        rw, rh, forward, inverse, _, _ = perspective_geometry(corners, a.arena_width_cm, a.arena_height_cm)
        _, samples = sample_frames(vp, total, 61)
        rect_samples = np.stack([cv2.warpPerspective(f, forward, (rw, rh)) for f in samples])
        background = np.percentile(rect_samples, 85, axis=0).astype(np.uint8)
        threshold, _ = robust_threshold(rect_samples, background, 0)

        print(f"[2/3] Scanning candidates of {vp.name} ...", flush=True)
        cap = cv2.VideoCapture(str(vp))
        good, junk = scan_candidates(cap, total, forward, rw, rh, background, threshold, 2000, set())
        picks = farthest_pick(np.stack([d for _, d, _ in good]), a.per_video)
        items = []
        for i in picks:
            f, _, contour = good[i]
            src = cv2.perspectiveTransform(contour.astype(np.float32), inverse)[:, 0, :]
            items.append({"frame": int(f), "contour": src, "junk": False})
        for f, _ in junk[: a.junk]:
            items.append({"frame": int(f), "contour": None, "junk": True})
        print(f"      {len(good)} good / {len(junk)} junk frames; showing {len(items)} candidates", flush=True)
        if a.dry_run:
            for it in items:
                rows.append({"video": vp.name, "frame": it["frame"], "exclude": int(it["junk"])})
            print(f"[dry-run] {len(items)} rows for {vp.name}")
            continue
        if not items:
            print(f"[3/3] No candidates for {vp.name}; skipping")
            continue
        excluded = montage_session(cap, items, vp.name)
        for i, it in enumerate(items):
            rows.append({"video": vp.name, "frame": it["frame"], "exclude": int(i in excluded)})
        cap.release()
        save_rows(Path(a.output), rows); rows = []
        print(f"[3/3] Done {vp.name}", flush=True)

    if a.dry_run:
        save_rows(Path(a.output), rows)
        print("Dry run complete.")


if __name__ == "__main__":
    main()