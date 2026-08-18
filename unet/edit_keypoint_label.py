#!/usr/bin/env python3
"""Edit one Head/Reflection annotation from the contact-sheet viewer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "traditional" / "code"))

from mouse_behavior_pipeline import perspective_geometry  # noqa: E402
from edit_polygon_label import create_editor_window, window_is_visible  # noqa: E402
from label_compat import as_bool, atomic_upsert_head, video_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--heads", required=True)
    parser.add_argument("--roi-json", required=True)
    parser.add_argument("--arena-width-cm", type=float, required=True)
    parser.add_argument("--arena-height-cm", type=float, required=True)
    parser.add_argument("--initial-head", default="")
    parser.add_argument("--initial-reflection", default="")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {args.video} frame {args.frame}")

    roi = json.loads(Path(args.roi_json).read_text(encoding="utf-8"))
    corners = np.asarray(roi["arena_corners_px"], np.float32)
    rw, rh, forward, inverse, _, _ = perspective_geometry(
        corners, args.arena_width_cm, args.arena_height_cm)

    def px_to_cm(point):
        rect = cv2.perspectiveTransform(np.asarray([[point]], np.float32), forward)[0, 0]
        return (float(rect[0] * args.arena_width_cm / (rw - 1)),
                float(rect[1] * args.arena_height_cm / (rh - 1)))

    def cm_to_px(point):
        if point is None or not np.isfinite(point).all():
            return None
        rect = np.asarray([[[point[0] / args.arena_width_cm * (rw - 1),
                             point[1] / args.arena_height_cm * (rh - 1)]]], np.float32)
        return tuple(cv2.perspectiveTransform(rect, inverse)[0, 0])

    state = {"active": "reflection", "head": None, "reflection": None,
             "head_verified": False, "reflection_verified": False,
             "exclude": False, "drag": False, "saved": 0, "message": "ready"}
    heads_path = Path(args.heads)
    if heads_path.is_file():
        try:
            rows = pd.read_csv(heads_path)
            matches = (video_mask(rows["video"], args.video) &
                       (pd.to_numeric(rows["frame"], errors="coerce") == args.frame))
            if matches.any():
                row = rows.loc[matches].iloc[-1]
                for prefix in ("head", "reflection"):
                    xy = pd.to_numeric(row[[f"{prefix}_x_cm", f"{prefix}_y_cm"]],
                                       errors="coerce").to_numpy(float)
                    state[prefix] = tuple(xy) if np.isfinite(xy).all() else None
                    state[f"{prefix}_verified"] = as_bool(row.get(f"{prefix}_verified", False))
                state["exclude"] = as_bool(row.get("exclude", False))
        except (OSError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    def heatmap_initial(path_value: str):
        path = Path(path_value)
        heatmap = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        if heatmap is None or heatmap.max() <= 0:
            return None
        _, _, _, point = cv2.minMaxLoc(heatmap)
        return px_to_cm((point[0] * image.shape[1] / heatmap.shape[1],
                         point[1] * image.shape[0] / heatmap.shape[0]))

    if state["head"] is None:
        state["head"] = heatmap_initial(args.initial_head)
    if state["reflection"] is None:
        state["reflection"] = heatmap_initial(args.initial_reflection)

    scale = min(2.0, 1400.0 / image.shape[1], 880.0 / image.shape[0])
    display_size = (max(1, int(round(image.shape[1] * scale))),
                    max(1, int(round(image.shape[0] * scale))))
    title = f"Head Reflection editor - frame {args.frame}"

    def save(reason: str) -> None:
        atomic_upsert_head(args.heads, args.video, args.frame, args.frame / fps,
                           state["head"], state["reflection"], state["exclude"],
                           source="contact_sheet_keypoint_edit",
                           head_verified=state["head_verified"],
                           reflection_verified=state["reflection_verified"])
        state["saved"] += 1
        state["message"] = f"AUTO-SAVED #{state['saved']} ({reason})"

    def mouse(event, x, y, _flags, _param):
        point = (x / scale, y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = True
            state[state["active"]] = px_to_cm(point)
            state[state["active"] + "_verified"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state[state["active"]] = px_to_cm(point)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            state["drag"] = False
            state[state["active"]] = px_to_cm(point)
            save(state["active"])

    def draw():
        shown = cv2.resize(image, display_size, interpolation=cv2.INTER_LINEAR)
        for key, color, label in (("head", (0, 0, 255), "HEAD"),
                                  ("reflection", (255, 0, 255), "REFLECTION")):
            point = cm_to_px(state[key])
            if point is not None:
                pxy = tuple(np.rint(np.asarray(point) * scale).astype(int))
                thickness = -1 if state[key + "_verified"] else 2
                cv2.circle(shown, pxy, 8, color, thickness, cv2.LINE_AA)
                cv2.putText(shown, label, (pxy[0] + 10, pxy[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 67), (0, 0, 0), -1)
        active_color = (255, 0, 255) if state["active"] == "reflection" else (0, 0, 255)
        cv2.putText(shown, f"ACTIVE={state['active'].upper()}  H=head  R=reflection  click/drag=move+verify",
                    (9, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, active_color, 1, cv2.LINE_AA)
        cv2.putText(shown, "C=confirm active  X=reflection absent  N=head absent  E=exclude  S=save+close  Q=close",
                    (9, 44), cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(shown, state["message"], (9, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    .37, (70, 230, 120), 1, cv2.LINE_AA)
        return shown

    create_editor_window(title, draw(), mouse)
    while True:
        if not window_is_visible(title):
            break
        cv2.imshow(title, draw())
        key = cv2.waitKey(20) & 255
        if key in (27, ord("q")):
            break
        if key == ord("h"):
            state["active"] = "head"
        elif key == ord("r"):
            state["active"] = "reflection"
        elif key == ord("c"):
            state[state["active"] + "_verified"] = True
            save("confirm " + state["active"])
        elif key == ord("x"):
            state["reflection"] = None
            state["reflection_verified"] = True
            save("reflection absent")
        elif key == ord("n"):
            state["head"] = None
            state["head_verified"] = True
            save("head absent")
        elif key == ord("e"):
            state["exclude"] = not state["exclude"]
            save("exclude")
        elif key == ord("s"):
            save("finish")
            break
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error:
        pass
    print(f"Saved frame {args.frame} {state['saved']} time(s) to {args.heads}")


if __name__ == "__main__":
    main()
