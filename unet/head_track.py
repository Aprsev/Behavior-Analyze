#!/usr/bin/env python3
"""Combine the U-Net mask with the miniscope reflection to track body + head.

For every frame:
  - the U-Net supplies a clean mouse+miniscope mask (fibre/tail excluded);
  - the body is the mask centroid (rectified arena coordinates -> cm);
  - the head is the brightest compact spot INSIDE the dilated U-Net mask
    (ReflectionTracker): the clean mask removes floor/fibre reflections that
    used to fool the dark-background pipeline;
  - excluded frames (--exclude-csv) become NaN rows and are marked in the
    overlay video.

Outputs: head_track_trajectory.csv, head_track_overlay.mp4,
head_track_metadata.json.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "traditional" / "code"
sys.path.insert(0, str(CODE))
from calibrate import detect_floor_bounds  # noqa: E402
from compare_head_methods import ReflectionTracker  # noqa: E402
from mouse_behavior_pipeline import (  # noqa: E402
    perspective_geometry, rectified_to_cm, robust_threshold, sample_frames, video_properties,
)
from model import UNet  # noqa: E402
from preprocess import estimate_background, bg_centered  # noqa: E402


def rot_pt(p: tuple[float, float], k: int, h: int, w: int) -> tuple[float, float]:
    """(col,row) in an (h,w) frame -> (col,row) in np.rot90(frame, k)."""
    x, y = p
    for _ in range(k % 4):
        x, y = y, w - 1 - x
        h, w = w, h
    return x, y


def inv_rot_pt(p: tuple[float, float], k: int, h: int, w: int) -> tuple[float, float]:
    """inverse of rot_pt (p is in the rotated frame)."""
    x, y = p
    for _ in range(k % 4):
        h, w = w, h
    for _ in range(k % 4):
        x, y = h - 1 - y, x
        h, w = w, h
    return x, y


def load_corners(roi_json: Path) -> np.ndarray:
    data = json.loads(Path(roi_json).read_text(encoding="utf-8"))
    return np.asarray(data["arena_corners_px"], np.float32)


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask), None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8) * 255, (float(cent[label][0]), float(cent[label][1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True); p.add_argument("--model", required=True)
    p.add_argument("--roi-json", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--arena-width-cm", type=float, default=25.0)
    p.add_argument("--arena-height-cm", type=float, default=30.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="arena turned vs training: rotate frames before the CNN "
                        "(ROI corners are rotated too; output stays in source space)")
    p.add_argument("--exclude-csv", default="")
    a = p.parse_args()

    excluded: set[int] = set()
    if a.exclude_csv and Path(a.exclude_csv).is_file():
        ex = pd.read_csv(a.exclude_csv)
        ex = ex.loc[ex.exclude.fillna(False).astype(bool) & (ex.video == Path(a.video).name)]
        excluded = set(int(f) for f in ex.frame)
        print(f"Marking {len(excluded)} screened frames as excluded")

    pack = torch.load(a.model, map_location="cpu")
    size = int(pack["size"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet().to(dev); net.load_state_dict(pack["state_dict"]); net.eval()

    corners = load_corners(Path(a.roi_json))
    cap = cv2.VideoCapture(a.video)
    w, h = int(cap.get(3)), int(cap.get(4))
    k = int(a.rotate) // 90 % 4
    if k:
        # Arena turned by a quarter-turn: rotate the frame for the CNN and
        # rotate the ROI corners the same way so perspective stays aligned.
        corners = np.asarray([rot_pt(tuple(c), k, h, w) for c in corners], np.float32)

    # Background-invariant CNN input, identical to train.py: active only
    # when the model was trained with it (old checkpoints keep raw frames).
    # This is a source-space background (the CNN never sees the rectified
    # one below, which serves the reflection tracker).
    bg_small = None
    if bool(pack.get("bg_subtract")):
        bg = estimate_background(Path(a.video))
        if bg is not None:
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            if k:
                bg_gray = np.rot90(bg_gray, k)  # same rotation as the frames
            bg_small = cv2.resize(bg_gray, (size, size), interpolation=cv2.INTER_AREA)
            print(f"Model trained with background subtraction - applying "
                  f"background-centered input (gain {pack.get('bg_gain', 2.0):.1f})")
        else:
            print("WARNING: background estimation failed; using raw frames")
    total, fps, _, _ = video_properties(Path(a.video))
    rw, rh, forward, inverse, _, _ = perspective_geometry(corners, a.arena_width_cm, a.arena_height_cm)
    _, samples = sample_frames(Path(a.video), total, 61)
    rect_samples = np.stack([cv2.warpPerspective(np.rot90(f, k) if k else f, forward, (rw, rh)) for f in samples])
    background = np.percentile(rect_samples, 85, axis=0).astype(np.uint8)
    robust_threshold(rect_samples, background, 0)

    # Wall-band exclusion: the ROI may include the walls (the camera sees
    # the mouse's shadow projected on them). Wall content changes with the
    # mouse position, so everything outside the detected floor rectangle is
    # zeroed in the mask and can never influence body/head positions.
    floor = detect_floor_bounds(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
    wall_mask = None
    if floor is not None:
        x0, y0, x1, y1 = floor
        wall_mask = np.ones((rh, rw), np.uint8) * 255
        wall_mask[y0:y1, x0:x1] = 0
        share = 100.0 * (x1 - x0) * (y1 - y0) / (rw * rh)
        print(f"Floor detected {floor} ({share:.0f}% of rectified arena); "
              f"wall band excluded from segmentation")
    else:
        print("No distinct wall band detected; whole rectified arena is floor")

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ow = cv2.VideoWriter(str(out / "head_track_overlay.mp4"), fourcc, fps, (w, h))
    tracker = ReflectionTracker(fps)
    rows = []; i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k:
            frame = np.rot90(frame, k)  # everything below works in rotated space
        if i in excluded:
            body = head = None; conf = 0.0; overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), (0, 0, 220), 6)
            cv2.putText(overlay, f"EXCLUDED frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)
            mask_src = None
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (size, size))
            if bg_small is not None:
                small = bg_centered(small, bg_small)
            x = torch.from_numpy(small[None, None].copy()).float().to(dev) / 255
            with torch.no_grad():
                prob = torch.sigmoid(net(x))[0, 0].cpu().numpy()
            mask_src = (cv2.resize(prob, (frame.shape[1], frame.shape[0])) >= a.threshold).astype(np.uint8) * 255
            mask_src, body_src = largest_component(mask_src)
            body = None; head = None; conf = 0.0
            if body_src is not None:
                body_mask = cv2.warpPerspective(mask_src, forward, (rw, rh), flags=cv2.INTER_NEAREST)
                if wall_mask is not None:
                    body_mask[wall_mask > 0] = 0  # wall projections excluded
                # body = centroid of the wall-excluded mask (rectified space)
                m = cv2.moments(body_mask)
                if m["m00"] > 0:
                    body = (m["m10"] / m["m00"], m["m01"] / m["m00"])
                    rect = cv2.warpPerspective(frame, forward, (rw, rh))
                    head, conf, status = tracker.update(rect, background, body, body_mask)
                    if head is not None:
                        head = tuple(float(v) for v in head)
                else:
                    body = None; head = None; conf = 0.0
            overlay = frame.copy()
            overlay[mask_src > 0] = (0, 220, 0)
            overlay = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
            if wall_mask is not None:
                # floor rectangle outline (rectified -> source space)
                fpts = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
                src_pts = cv2.perspectiveTransform(fpts[None], inverse)[0]
                cv2.polylines(overlay, [src_pts.astype(np.int32)], True, (255, 128, 0), 2, cv2.LINE_AA)
            if body is not None:
                b_px = cv2.perspectiveTransform(np.asarray([[body]], np.float32), inverse)[0, 0].astype(int)
                cv2.circle(overlay, tuple(b_px), 6, (0, 0, 255), -1, cv2.LINE_AA)
                if head is not None:
                    h_px = cv2.perspectiveTransform(np.asarray([[head]], np.float32), inverse)[0, 0].astype(int)
                    cv2.circle(overlay, tuple(h_px), 7, (0, 220, 255), -1, cv2.LINE_AA)
                    cv2.line(overlay, tuple(b_px), tuple(h_px), (255, 255, 255), 1, cv2.LINE_AA)
        ow.write(np.rot90(overlay, -k) if k else overlay)
        body_cm = rectified_to_cm(body, rw, rh, a.arena_width_cm, a.arena_height_cm) if body else (float("nan"), float("nan"))
        head_cm = rectified_to_cm(head, rw, rh, a.arena_width_cm, a.arena_height_cm) if head else (float("nan"), float("nan"))
        rows.append((i, i / fps, body_cm[0], body_cm[1], head_cm[0], head_cm[1], conf))
        i += 1
    cap.release(); ow.release()
    df = pd.DataFrame(rows, columns=["frame", "timestamp_sec", "body_x_cm", "body_y_cm",
                                     "head_x_cm", "head_y_cm", "head_confidence"])
    df.to_csv(out / "head_track_trajectory.csv", index=False, float_format="%.5f")
    (out / "head_track_metadata.json").write_text(json.dumps(
        {"device": dev, "frames": i, "threshold": a.threshold, "rotate": a.rotate,
         "bg_subtract": bg_small is not None,
         "floor_bounds": list(floor) if floor is not None else None, "model_size": size,
         "head_valid_percent": round(100 * float(df.head_x_cm.notna().mean()), 2),
         "excluded_frames": sorted(excluded & set(range(i)))}, indent=2), encoding="utf-8")
    print(f"Wrote {i} frames to {out}; head valid {df.head_x_cm.notna().mean():.1%}")


if __name__ == "__main__":
    main()