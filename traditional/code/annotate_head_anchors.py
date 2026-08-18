#!/usr/bin/env python3
"""Manually correct head and miniscope-reflection anchors for lightweight training.

This program intentionally collects two different labels:
  * head: the anatomical head/nose position (red)
  * reflection: the visible miniscope reflection (magenta), if present

They must not be conflated: a reflection can be consistently offset from the
head, can disappear, or can be uninformative in particular poses.  An excluded
frame is retained in the CSV but is not used as a head-training example.

Mouse controls
--------------
Click or drag RED to set the anatomical head. Click or drag MAGENTA to set the
reflection anchor. The active target can be changed with H/R.

Keyboard controls
-----------------
H: edit head (red)       R: edit reflection (magenta)   C: confirm active point
E: toggle exclusion      X: mark reflection absent
N: mark head absent      A/D or left/right: previous/next candidate frame
J/L: -10/+10 frames      S: save now                 Q/Esc: save and quit

The defaults select a balanced sample of uncertain, disagreeing, and regular
frames. Use --all-frames only when you explicitly want sequential annotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


HEAD_COLUMNS = ("head_silhouette_x_cm", "head_silhouette_y_cm")
REFLECTION_COLUMNS = ("head_reflection_x_cm", "head_reflection_y_cm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Original video")
    parser.add_argument("--comparison-csv", required=True, help="head_method_comparison.csv")
    parser.add_argument("--roi-json", required=True, help="arena_roi.json used for the automatic comparison")
    parser.add_argument("--output", default="manual_head_anchor_labels.csv")
    parser.add_argument("--arena-width-cm", type=float, required=True)
    parser.add_argument("--arena-height-cm", type=float, required=True)
    parser.add_argument("--max-labels", type=int, default=100, help="Number of informative frames to present")
    parser.add_argument("--all-frames", action="store_true", help="Annotate every frame in original order")
    return parser.parse_args()


def load_existing(path: Path) -> pd.DataFrame:
    columns = ["frame", "timestamp_sec", "head_x_cm", "head_y_cm", "reflection_x_cm", "reflection_y_cm", "exclude", "reflection_present", "head_present", "head_verified", "reflection_verified"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    data = pd.read_csv(path)
    for column in columns:
        if column not in data:
            data[column] = np.nan
    return data[columns]


def choose_frames(data: pd.DataFrame, maximum: int, all_frames: bool) -> list[int]:
    if all_frames:
        return data.frame.astype(int).tolist()
    head = data.loc[:, HEAD_COLUMNS].to_numpy(float)
    reflection = data.loc[:, REFLECTION_COLUMNS].to_numpy(float)
    both = np.isfinite(head).all(axis=1) & np.isfinite(reflection).all(axis=1)
    disagreement = np.full(len(data), 99.0)
    disagreement[both] = np.linalg.norm(head[both] - reflection[both], axis=1)
    confidence = data.get("reflection_confidence", pd.Series(0, index=data.index)).fillna(0).to_numpy(float)
    reflection_missing = ~np.isfinite(reflection).all(axis=1)
    # Prioritise ambiguity/failures, then distribute the remaining frames over
    # the complete recording so that a classifier is not trained on one pose.
    priority = np.argsort(-(reflection_missing * 5 + (1 - confidence) * 2 + np.clip(disagreement, 0, 8) / 4))
    selected: list[int] = []
    seen: set[int] = set()
    for index in priority:
        frame = int(data.frame.iloc[index])
        if frame not in seen:
            selected.append(frame); seen.add(frame)
        if len(selected) >= maximum // 2:
            break
    grid = np.linspace(0, len(data) - 1, max(1, maximum - len(selected)), dtype=int)
    for index in grid:
        frame = int(data.frame.iloc[index])
        if frame not in seen:
            selected.append(frame); seen.add(frame)
    return selected[:maximum]


class Annotator:
    def __init__(self, args: argparse.Namespace, source: pd.DataFrame, labels: pd.DataFrame, frames: list[int]):
        self.args, self.source, self.labels, self.frames = args, source.set_index("frame"), labels.set_index("frame", drop=False), frames
        self.index = 0
        self.active = "head"
        self.dragging = False
        self.cap = cv2.VideoCapture(args.input)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.title = "Manual head / reflection labels"
        corners = np.asarray(__import__("json").loads(Path(args.roi_json).read_text(encoding="utf-8"))["arena_corners_px"], np.float32)
        rect_w, rect_h = round(args.arena_width_cm * 10), round(args.arena_height_cm * 10)
        dst = np.asarray([[0, 0], [rect_w - 1, 0], [rect_w - 1, rect_h - 1], [0, rect_h - 1]], np.float32)
        self.forward = cv2.getPerspectiveTransform(corners, dst)
        self.inverse = cv2.getPerspectiveTransform(dst, corners)
        self.rect_w, self.rect_h = rect_w, rect_h

    def cm_to_px(self, xy: tuple[float, float]) -> tuple[int, int] | None:
        if not np.isfinite(xy).all(): return None
        rect = np.asarray([[[xy[0] / self.args.arena_width_cm * (self.rect_w - 1), xy[1] / self.args.arena_height_cm * (self.rect_h - 1)]]], np.float32)
        pixel = cv2.perspectiveTransform(rect, self.inverse)[0, 0]
        return int(round(pixel[0])), int(round(pixel[1]))

    def px_to_cm(self, xy: tuple[int, int]) -> tuple[float, float]:
        rect = cv2.perspectiveTransform(np.asarray([[[xy[0], xy[1]]]], np.float32), self.forward)[0, 0]
        return rect[0] * self.args.arena_width_cm / (self.rect_w - 1), rect[1] * self.args.arena_height_cm / (self.rect_h - 1)

    def current(self) -> pd.Series:
        frame = self.frames[self.index]
        if frame not in self.labels.index:
            base = self.source.loc[frame]
            # A source row also includes string status columns, so Pandas may
            # expose the selected numeric entries as object dtype. Convert
            # explicitly before checking for missing automatic candidates.
            head_values = pd.to_numeric(base.loc[list(HEAD_COLUMNS)], errors="coerce").to_numpy(dtype=float)
            reflection_values = pd.to_numeric(base.loc[list(REFLECTION_COLUMNS)], errors="coerce").to_numpy(dtype=float)
            row = {"frame": frame, "timestamp_sec": float(base.timestamp_sec), "head_x_cm": head_values[0], "head_y_cm": head_values[1], "reflection_x_cm": reflection_values[0], "reflection_y_cm": reflection_values[1], "exclude": False, "reflection_present":bool(np.isfinite(reflection_values).all()), "head_present": bool(np.isfinite(head_values).all()), "head_verified": False, "reflection_verified": False}
            self.labels.loc[frame] = row
        return self.labels.loc[frame]

    def set_active_point(self, x: int, y: int) -> None:
        row = self.current()
        x_cm, y_cm = self.px_to_cm((x, y))
        if self.active == "head":
            self.labels.loc[row.frame, ["head_x_cm", "head_y_cm", "head_present", "head_verified"]] = [x_cm, y_cm, True, True]
        else:
            self.labels.loc[row.frame, ["reflection_x_cm", "reflection_y_cm", "reflection_present", "reflection_verified"]] = [x_cm, y_cm, True, True]

    def mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True; self.set_active_point(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.set_active_point(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False; self.set_active_point(x, y)

    def draw(self) -> np.ndarray:
        row = self.current()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame))
        ok, image = self.cap.read()
        if not ok: raise RuntimeError(f"Cannot read frame {row.frame}")
        head = self.cm_to_px((row.head_x_cm, row.head_y_cm)) if bool(row.head_present) else None
        reflection = self.cm_to_px((row.reflection_x_cm, row.reflection_y_cm)) if bool(row.reflection_present) else None
        for point, color, label in ((head, (0, 0, 255), "HEAD"), (reflection, (255, 0, 255), "REFLECTION")):
            if point:
                cv2.circle(image, point, 7, color, -1, cv2.LINE_AA)
                cv2.putText(image, label, (point[0] + 9, point[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
        state = "EXCLUDED" if bool(row.exclude) else f"edit={self.active.upper()}"
        message = f"{self.index + 1}/{len(self.frames)}  frame={int(row.frame)} t={row.timestamp_sec:.2f}s  {state}"
        cv2.rectangle(image, (0, 0), (self.width, 58), (0, 0, 0), -1)
        cv2.putText(image, message, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .56, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "H/R target  click-drag  E exclude  X reflect absent  N head absent  A/D step  J/L +/-10  S save  Q quit", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def save(self) -> None:
        result = self.labels.reset_index(drop=True).sort_values("frame")
        result["exclude"] = result.exclude.fillna(False).astype(bool)
        result["reflection_present"] = result.reflection_present.fillna(False).astype(bool)
        result["head_present"] = result.head_present.fillna(False).astype(bool)
        result["head_verified"] = result.head_verified.fillna(False).astype(bool)
        result["reflection_verified"] = result.reflection_verified.fillna(False).astype(bool)
        result.to_csv(self.args.output, index=False, float_format="%.6f")
        print(f"Saved {len(result)} labelled frames to {self.args.output}")

    def run(self) -> None:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL); cv2.setMouseCallback(self.title, self.mouse)
        while True:
            cv2.imshow(self.title, self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                self.save(); break
            if key == ord("h"): self.active = "head"
            elif key == ord("r"): self.active = "reflection"
            elif key == ord("c"):
                row = self.current(); self.labels.loc[row.frame, self.active + "_verified"] = True
            elif key == ord("e"):
                row = self.current(); self.labels.loc[row.frame, "exclude"] = not bool(row.exclude)
            elif key == ord("x"):
                row = self.current(); self.labels.loc[row.frame, ["reflection_present", "reflection_x_cm", "reflection_y_cm", "reflection_verified"]] = [False, np.nan, np.nan, True]
            elif key == ord("n"):
                row = self.current(); self.labels.loc[row.frame, ["head_present", "head_x_cm", "head_y_cm", "head_verified"]] = [False, np.nan, np.nan, True]
            elif key == ord("s"): self.save()
            elif key in (81, ord("a")): self.index = max(0, self.index - 1)
            elif key in (83, ord("d")): self.index = min(len(self.frames) - 1, self.index + 1)
            elif key == ord("j"): self.index = max(0, self.index - 10)
            elif key == ord("l"): self.index = min(len(self.frames) - 1, self.index + 10)
        self.cap.release(); cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.comparison_csv)
    required = {"frame", "timestamp_sec", *HEAD_COLUMNS, *REFLECTION_COLUMNS}
    if not required.issubset(source.columns): raise ValueError("comparison CSV is missing required head-method columns")
    labels = load_existing(Path(args.output))
    frames = choose_frames(source, args.max_labels, args.all_frames)
    Annotator(args, source, labels, frames).run()


if __name__ == "__main__":
    main()
