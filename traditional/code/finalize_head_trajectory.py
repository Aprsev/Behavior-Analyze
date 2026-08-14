#!/usr/bin/env python3
"""Apply a manual-label-trained calibrator and write final fused head output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd

from train_head_calibrator import FEATURES, feature_table


def regularize_head_trajectory(raw: np.ndarray, body: np.ndarray, fps: float, max_gap_seconds: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Reject implausible frame jumps, fill short gaps, then smooth segments.

    The limit follows body movement, so normal rapid running is retained while
    a one-frame candidate switch is removed. Long unavailable periods stay NaN
    rather than creating a fabricated trajectory.
    """
    values = raw.astype(float).copy()
    valid_body = np.isfinite(body).all(axis=1)
    values[~valid_body] = np.nan
    status = np.full(len(values), "model", dtype=object)
    status[~np.isfinite(values).all(axis=1)] = "missing"
    last_valid: int | None = None
    for index in range(len(values)):
        if not np.isfinite(values[index]).all():
            continue
        if last_valid is not None:
            step = float(np.linalg.norm(values[index] - values[last_valid]))
            body_step = float(np.linalg.norm(body[index] - body[last_valid])) if valid_body[index] and valid_body[last_valid] else 0.0
            elapsed = max(1, index - last_valid)
            # 55 cm/s maximum head speed plus body displacement; at 30 fps
            # this allows ~1.8 cm between adjacent frames and more after gaps.
            allowed = body_step + 55.0 * elapsed / max(fps, 1.0) + 0.35
            if step > allowed:
                values[index] = np.nan
                status[index] = "rejected_jump"
                continue
        last_valid = index
    table = pd.DataFrame(values, columns=["x", "y"])
    original_valid = table.notna().all(axis=1).to_numpy()
    max_gap = max(1, int(round(max_gap_seconds * fps)))
    table = table.interpolate(limit=max_gap, limit_area="inside")
    interpolated = ~original_valid & table.notna().all(axis=1).to_numpy()
    status[interpolated] = "interpolated"
    # Smooth within continuous spans only.  This avoids joining separate bouts
    # and gives the displayed head point a physically continuous curve.
    valid = table.notna().all(axis=1).to_numpy()
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    window = int(np.clip(round(fps * 0.17), 5, 15)) | 1
    try:
        from scipy.signal import savgol_filter
        for start, stop in zip(starts, stops):
            length = stop - start
            local = min(window, length if length % 2 else length - 1)
            if local >= 5:
                for column in ("x", "y"):
                    table.loc[start:stop - 1, column] = savgol_filter(table.loc[start:stop - 1, column].to_numpy(), local, 2, mode="interp")
    except ImportError:
        table = table.rolling(5, center=True, min_periods=1).mean().where(valid[:, None])
    # Pandas may expose a read-only view here; the caller needs to restore
    # manually labelled anchors exactly after smoothing.
    return table.to_numpy(copy=True), status


def draw_final_video(input_path: str, roi_json: str, trajectory: pd.DataFrame, arena_w: float, arena_h: float, output_path: str) -> None:
    """Render final calibrated head point in original camera coordinates."""
    metadata = json.loads(Path(roi_json).read_text(encoding="utf-8"))
    corners = np.asarray(metadata["arena_corners_px"], np.float32)
    rect_w, rect_h = round(arena_w * 10), round(arena_h * 10)
    destination = np.asarray([[0, 0], [rect_w - 1, 0], [rect_w - 1, rect_h - 1], [0, rect_h - 1]], np.float32)
    inverse = cv2.getPerspectiveTransform(destination, corners)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {output_path}")
    trail = np.zeros((height, width, 3), np.uint8)
    previous_head = None
    def point(x, y):
        if not np.isfinite([x, y]).all(): return None
        rect = np.asarray([[[x / arena_w * (rect_w - 1), y / arena_h * (rect_h - 1)]]], np.float32)
        src = cv2.perspectiveTransform(rect, inverse)[0, 0]
        return int(round(src[0])), int(round(src[1]))
    for _, row in trajectory.iterrows():
        ok, frame = cap.read()
        if not ok: break
        final_head = point(row.head_x_cm, row.head_y_cm)
        body = point(row.body_x_cm, row.body_y_cm)
        silhouette = point(row.head_silhouette_x_cm, row.head_silhouette_y_cm)
        reflection = point(row.head_reflection_x_cm, row.head_reflection_y_cm)
        if final_head is not None:
            if previous_head is not None: cv2.line(trail, previous_head, final_head, (0, 220, 0), 2, cv2.LINE_AA)
            previous_head = final_head
        else: previous_head = None
        cv2.addWeighted(frame, 1.0, trail, .7, 0, frame)
        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        if body: cv2.circle(frame, body, 4, (0, 220, 255), -1, cv2.LINE_AA)
        if silhouette: cv2.circle(frame, silhouette, 3, (0, 0, 255), -1, cv2.LINE_AA)
        if reflection: cv2.circle(frame, reflection, 3, (255, 0, 255), -1, cv2.LINE_AA)
        if final_head:
            cv2.circle(frame, final_head, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(frame, "final calibrated head", (final_head[0] + 8, final_head[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"frame {int(row.frame)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release(); writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", default="", help="Original video; enables final annotated video")
    parser.add_argument("--roi-json", default="", help="Arena ROI JSON; required with --input")
    parser.add_argument("--arena-width-cm", type=float, default=0)
    parser.add_argument("--arena-height-cm", type=float, default=0)
    parser.add_argument("--output-video", default="", help="Final annotated MP4")
    parser.add_argument("--max-gap-sec", type=float, default=0.5, help="Only interpolate missing head points up to this duration")
    parser.add_argument("--labels", default="", help="Manual label CSV; valid manual head labels are exact final-frame overrides")
    args = parser.parse_args()
    data = feature_table(pd.read_csv(args.comparison_csv))
    package = joblib.load(args.model)
    prediction = package["model"].predict(data[package.get("features", FEATURES)])
    if package.get("target") == "head_offset_from_body_cm":
        prediction += data[["body_x_cm", "body_y_cm"]].to_numpy(float)
    # No body detection means neither prediction nor confidence is meaningful.
    valid_body = data[["body_x_cm", "body_y_cm"]].notna().all(axis=1).to_numpy()
    prediction[~valid_body] = np.nan
    timestamps = data.timestamp_sec.to_numpy(float)
    fps = 1.0 / np.median(np.diff(timestamps)) if len(timestamps) > 1 else 30.0
    manual_override = np.zeros(len(data), dtype=bool)
    if args.labels:
        labels = pd.read_csv(args.labels)
        valid_labels = labels.loc[
            labels.head_present.fillna(False).astype(bool)
            & ~labels.exclude.fillna(False).astype(bool)
            & labels[["head_x_cm", "head_y_cm"]].notna().all(axis=1),
            ["frame", "head_x_cm", "head_y_cm"],
        ].set_index("frame")
        for index, frame in enumerate(data.frame.astype(int)):
            if frame in valid_labels.index:
                prediction[index] = valid_labels.loc[frame, ["head_x_cm", "head_y_cm"]].to_numpy(float)
                manual_override[index] = True
    regularized, regularization_status = regularize_head_trajectory(
        prediction, data[["body_x_cm", "body_y_cm"]].to_numpy(float), fps, args.max_gap_sec
    )
    # Preserve human-labelled anatomical points exactly after smoothing.
    regularized[manual_override] = prediction[manual_override]
    regularization_status[manual_override] = "manual_override"
    output = data[["frame", "timestamp_sec", "body_x_cm", "body_y_cm", "head_silhouette_x_cm", "head_silhouette_y_cm", "head_reflection_x_cm", "head_reflection_y_cm", "reflection_confidence"]].copy()
    output["head_raw_model_x_cm"] = prediction[:, 0]
    output["head_raw_model_y_cm"] = prediction[:, 1]
    output["head_x_cm"] = regularized[:, 0]
    output["head_y_cm"] = regularized[:, 1]
    output["head_regularization_status"] = regularization_status
    output["head_method"] = "manual_calibrated_fusion_smoothed"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Wrote {len(output)} frames to {args.output}")
    print(pd.Series(regularization_status).value_counts().to_string())
    if args.input:
        if not args.roi_json or args.arena_width_cm <= 0 or args.arena_height_cm <= 0:
            raise ValueError("--input requires --roi-json, --arena-width-cm, and --arena-height-cm")
        video = args.output_video or str(Path(args.output).with_name("annotated_manual_calibrated.mp4"))
        draw_final_video(args.input, args.roi_json, output, args.arena_width_cm, args.arena_height_cm, video)
        print(f"Wrote final annotated video to {video}")


if __name__ == "__main__":
    main()
