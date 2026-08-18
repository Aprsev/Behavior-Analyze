#!/usr/bin/env python3
"""Correct low-confidence head/reflection results selected from a finished run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "traditional" / "code"
sys.path.insert(0, str(CODE))
from mouse_behavior_pipeline import perspective_geometry, video_properties  # noqa: E402
from edit_polygon_label import create_editor_window, window_is_visible  # noqa: E402
from label_compat import (as_bool, atomic_upsert_head, atomic_upsert_polygon,
                          video_mask)  # noqa: E402


def rot_pt(point, k: int, height: int, width: int):
    x, y = point
    for _ in range(k % 4):
        x, y = y, width - 1 - x
        height, width = width, height
    return x, y


def select_frames(data: pd.DataFrame, existing: set[int], maximum: int) -> list[int]:
    confidence_col = data["head_confidence"] if "head_confidence" in data else pd.Series(0, index=data.index)
    disagreement_col = data["head_disagreement_px"] if "head_disagreement_px" in data else pd.Series(0, index=data.index)
    confidence = pd.to_numeric(confidence_col, errors="coerce").fillna(0).to_numpy()
    disagreement = pd.to_numeric(disagreement_col, errors="coerce").fillna(0).to_numpy()
    source = data.get("head_source", pd.Series("missing", index=data.index)).astype(str)
    fallback = source.isin(["learned_fallback", "missing"]).to_numpy(float)
    head = data[["head_x_cm", "head_y_cm"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    jumps = np.r_[0.0, np.linalg.norm(np.diff(head, axis=0), axis=1)]
    jumps[~np.isfinite(jumps)] = 0
    score = 4 * fallback + 2 * (1 - np.clip(confidence, 0, 1)) + np.clip(disagreement / 12, 0, 4) + np.clip(jumps / 2, 0, 4)
    candidates = [(float(score[i]), int(data.frame.iloc[i])) for i in range(len(data))
                  if int(data.frame.iloc[i]) not in existing and source.iloc[i] != "excluded"]
    candidates.sort(reverse=True)
    return [frame for _, frame in candidates[:maximum]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True); p.add_argument("--trajectory", required=True)
    p.add_argument("--roi-json", required=True); p.add_argument("--heads", required=True)
    p.add_argument("--mask-video", default="")
    p.add_argument("--torso-labels", default="")
    p.add_argument("--arena-width-cm", type=float, default=25.0)
    p.add_argument("--arena-height-cm", type=float, default=30.0)
    p.add_argument("--max-labels", type=int, default=20)
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    a = p.parse_args()

    data = pd.read_csv(a.trajectory).drop_duplicates("frame", keep="last").set_index("frame", drop=False)
    existing = set()
    if Path(a.heads).is_file():
        old = pd.read_csv(a.heads)
        if "video" in old:
            old = old.loc[video_mask(old.video, Path(a.video).name)]
        existing = set(pd.to_numeric(old.frame, errors="coerce").dropna().astype(int)) if "frame" in old else set()
    frames = select_frames(data.reset_index(drop=True), existing, a.max_labels)
    if not frames:
        print("No new low-confidence head frames remain to annotate.")
        return

    roi = json.loads(Path(a.roi_json).read_text(encoding="utf-8"))
    corners = np.asarray(roi["arena_corners_px"], np.float32)
    cap = cv2.VideoCapture(a.video)
    mask_cap = cv2.VideoCapture(a.mask_video) if a.mask_video and Path(a.mask_video).is_file() else None
    width, height = int(cap.get(3)), int(cap.get(4))
    total, fps, _, _ = video_properties(Path(a.video))
    k = a.rotate // 90 % 4
    if k:
        corners = np.asarray([rot_pt(tuple(point), k, height, width) for point in corners], np.float32)
    rw, rh, forward, inverse, _, _ = perspective_geometry(
        corners, a.arena_width_cm, a.arena_height_cm)
    state = {"i": 0, "active": "head", "head": None, "reflection": None,
             "drag": False, "exclude": False, "saved": 0}
    title = "Head result correction"

    def cm_to_px(point):
        if point is None or not np.isfinite(point).all():
            return None
        rect = np.asarray([[[point[0] / a.arena_width_cm * (rw - 1),
                             point[1] / a.arena_height_cm * (rh - 1)]]], np.float32)
        return tuple(cv2.perspectiveTransform(rect, inverse)[0, 0])

    def px_to_cm(point):
        rect = cv2.perspectiveTransform(np.asarray([[point]], np.float32), forward)[0, 0]
        return (float(rect[0] * a.arena_width_cm / (rw - 1)),
                float(rect[1] * a.arena_height_cm / (rh - 1)))

    def load():
        row = data.loc[frames[state["i"]]]
        def point(prefix):
            x = pd.to_numeric(row.get(f"{prefix}_x_cm"), errors="coerce")
            y = pd.to_numeric(row.get(f"{prefix}_y_cm"), errors="coerce")
            return (float(x), float(y)) if np.isfinite([x, y]).all() else None
        state["head"] = point("head")
        state["reflection"] = point("reflection")
        state["exclude"] = False

    def frame_image():
        frame = frames[state["i"]]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
        if not ok: raise RuntimeError(f"Cannot read frame {frame}")
        return np.rot90(image, k).copy() if k else image

    def draw():
        image = frame_image(); frame = frames[state["i"]]; row = data.loc[frame]
        for key, color, label in (("head", (0, 0, 255), "HEAD"),
                                  ("reflection", (255, 0, 255), "REFLECTION")):
            point = cm_to_px(state[key])
            if point is not None:
                pxy = tuple(np.rint(point).astype(int)); cv2.circle(image, pxy, 8, color, -1, cv2.LINE_AA)
                cv2.putText(image, label, (pxy[0] + 9, pxy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        cv2.rectangle(image, (0, 0), (image.shape[1], 58), (0, 0, 0), -1)
        cv2.putText(image, f"{state['i']+1}/{len(frames)} frame={frame} source={row.get('head_source','?')} active={state['active'].upper()}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
        cv2.putText(image, "H/R select  drag point  X reflection absent  N head absent  E exclude  A/D  S save+next  Q save+quit",
                    (8, 45), cv2.FONT_HERSHEY_SIMPLEX, .37, (255, 255, 255), 1)
        return image

    def save():
        frame = frames[state["i"]]
        atomic_upsert_head(a.heads, a.video, frame, frame / fps,
                           state["head"], state["reflection"], state["exclude"])
        # The green mask has already been accepted as accurate. Snapshot it as
        # this frame's torso label so every newly corrected head point is
        # actually exportable as a supervised heatmap during the next rebuild.
        if mask_cap is not None and a.torso_labels:
            mask_cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, mask_frame = mask_cap.read()
            if ok:
                gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                contours, _ = cv2.findContours((gray > 127).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    polygon = cv2.approxPolyDP(contour, .005 * cv2.arcLength(contour, True), True).reshape(-1, 2)
                    atomic_upsert_polygon(a.torso_labels, a.video, frame, polygon,
                                          state["exclude"], "head_result_mask_snapshot")
        state["saved"] += 1

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = True; state[state["active"]] = px_to_cm((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state[state["active"]] = px_to_cm((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = False; state[state["active"]] = px_to_cm((x, y))

    load(); create_editor_window(title, draw(), mouse)
    while True:
        if not window_is_visible(title):
            save(); break
        cv2.imshow(title, draw()); key = cv2.waitKey(20) & 255
        if key in (27, ord("q")):
            save(); break
        if key == ord("h"): state["active"] = "head"
        elif key == ord("r"): state["active"] = "reflection"
        elif key == ord("x"): state["reflection"] = None
        elif key == ord("n"): state["head"] = None
        elif key == ord("e"): state["exclude"] = not state["exclude"]
        elif key == ord("s"):
            save()
            if state["i"] < len(frames) - 1: state["i"] += 1; load()
            else: break
        elif key in (81, ord("a")): state["i"] = max(0, state["i"] - 1); load()
        elif key in (83, ord("d")): state["i"] = min(len(frames) - 1, state["i"] + 1); load()
    cap.release()
    if mask_cap is not None: mask_cap.release()
    try: cv2.destroyAllWindows(); cv2.waitKey(1)
    except cv2.error: pass
    print(f"Saved {state['saved']} head/reflection corrections to {a.heads}")


if __name__ == "__main__":
    main()
