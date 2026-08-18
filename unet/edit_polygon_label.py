#!/usr/bin/env python3
"""Large single-frame polygon editor launched from the contact sheet."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from label_compat import as_bool, atomic_upsert_polygon, normalize_polygon, video_mask


def simplify(points: np.ndarray, count: int = 8) -> np.ndarray:
    if len(points) <= count:
        return points.astype(float)
    return points[np.linspace(0, len(points) - 1, count, dtype=int)].astype(float)


def create_editor_window(title: str, preview: np.ndarray, mouse_callback) -> None:
    """Create HighGUI eagerly before registering callbacks.

    OpenCV 5's Qt backend on Windows may not allocate the native window until
    the first imshow/waitKey. Calling setMouseCallback immediately after
    namedWindow then raises ``NULL window handler``. Try two portable modes and
    register the callback only after a frame has been presented.
    """
    errors = []
    for mode in (cv2.WINDOW_NORMAL, cv2.WINDOW_AUTOSIZE):
        try:
            cv2.namedWindow(title, mode)
            cv2.imshow(title, preview)
            cv2.waitKey(50)
            if mode == cv2.WINDOW_NORMAL:
                cv2.resizeWindow(title, preview.shape[1], preview.shape[0])
            cv2.setMouseCallback(title, mouse_callback)
            return
        except cv2.error as exc:
            errors.append(str(exc))
            try:
                cv2.destroyWindow(title)
                cv2.waitKey(1)
            except cv2.error:
                pass
    raise RuntimeError("OpenCV could not create an interactive editor window. " + " | ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--initial-mask", default="")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, image = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {args.video} frame {args.frame}")

    points = None; excluded = False
    labels_path = Path(args.labels)
    if labels_path.is_file():
        rows = pd.read_csv(labels_path)
        if "video" in rows and "frame" in rows:
            matches = video_mask(rows.video, args.video) & (pd.to_numeric(rows.frame, errors="coerce") == args.frame)
            if matches.any():
                row = rows.loc[matches].iloc[-1]
                points = normalize_polygon(row.polygon_px).astype(float)
                excluded = as_bool(row.get("exclude", False))
    if points is None and args.initial_mask:
        mask = cv2.imread(args.initial_mask, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            if contours:
                points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
                points[:, 0] *= image.shape[1] / mask.shape[1]
                points[:, 1] *= image.shape[0] / mask.shape[0]
                points = simplify(points, 16)
    if points is None:
        raise RuntimeError("No editable polygon was found for this sample")

    initial = points.copy()
    scale = min(2.0, 1400.0 / image.shape[1], 880.0 / image.shape[0])
    display_size = (max(1, int(round(image.shape[1] * scale))),
                    max(1, int(round(image.shape[0] * scale))))
    state = {"points": points, "drag": None, "last": tuple(points[0]),
             "excluded": excluded, "saved": 0, "message": "ready"}
    # ASCII-only title avoids a second OpenCV 5/Qt handle issue on some
    # Windows locales; the video/frame identity is also drawn inside the UI.
    title = f"Polygon label editor - frame {args.frame}"

    def nearest(x: float, y: float) -> tuple[int, float]:
        distance = np.linalg.norm(state["points"] - np.asarray([x, y]), axis=1)
        index = int(distance.argmin())
        return index, float(distance[index])

    def save() -> None:
        backup = atomic_upsert_polygon(args.labels, args.video, args.frame,
                                       state["points"], state["excluded"])
        state["saved"] += 1
        state["message"] = f"AUTO-SAVED #{state['saved']} · backup {backup.name}"

    def mouse(event, x, y, _flags, _param):
        x, y = x / scale, y / scale
        state["last"] = (x, y)
        points_ = state["points"]
        if event == cv2.EVENT_LBUTTONDOWN:
            index, distance = nearest(x, y)
            if distance < 18 / scale:
                state["drag"] = index
            else:
                edges = np.roll(points_, -1, axis=0) - points_
                q = np.asarray([x, y]) - points_
                t = np.clip(np.sum(q * edges, axis=1) /
                            np.maximum(np.sum(edges * edges, axis=1), 1e-6), 0, 1)
                distance_to_edge = np.linalg.norm(q - edges * t[:, None], axis=1)
                index = int(distance_to_edge.argmin()) + 1
                state["points"] = np.insert(points_, index, [x, y], axis=0)
                state["drag"] = index
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            state["points"][state["drag"]] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and state["drag"] is not None:
            state["points"][state["drag"]] = [x, y]
            state["drag"] = None
            save()
        elif event == cv2.EVENT_RBUTTONDOWN and len(points_) > 3:
            index, _ = nearest(x, y)
            state["points"] = np.delete(points_, index, axis=0)
            save()

    first_preview = cv2.resize(image, display_size, interpolation=cv2.INTER_LINEAR)
    create_editor_window(title, first_preview, mouse)
    while True:
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            save(); break
        shown = cv2.resize(image, display_size, interpolation=cv2.INTER_LINEAR)
        polygon = np.rint(state["points"] * scale).astype(np.int32)
        layer = shown.copy(); cv2.fillPoly(layer, [polygon], (25, 190, 65))
        shown = cv2.addWeighted(shown, .78, layer, .22, 0)
        color = (0, 0, 255) if state["excluded"] else (255, 255, 0)
        cv2.polylines(shown, [polygon], True, color, 2, cv2.LINE_AA)
        for x, y in polygon:
            cv2.circle(shown, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 66), (0, 0, 0), -1)
        cv2.putText(shown, "Drag point / click edge add / right-click delete / T thin / R reset / E exclude",
                    (9, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(shown, f"S save+close / Q close  |  {state['message']}"
                    f"  |  points={len(state['points'])} exclude={state['excluded']}",
                    (9, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        cv2.imshow(title, shown)
        key = cv2.waitKey(20) & 255
        if key == ord("s"):
            save(); break
        if key in (27, ord("q")):
            break
        if key == ord("t"):
            state["points"] = simplify(state["points"], 8); save()
        elif key == ord("r"):
            state["points"] = initial.copy(); save()
        elif key in (ord("e"), ord("E")):
            state["excluded"] = not state["excluded"]; save()
        elif key in (8, 127) and len(state["points"]) > 3:
            index, _ = nearest(*state["last"])
            state["points"] = np.delete(state["points"], index, axis=0); save()
    cv2.destroyAllWindows()
    print(f"Saved frame {args.frame} {state['saved']} time(s) to {args.labels}")


if __name__ == "__main__":
    main()
