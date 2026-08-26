"""Hybrid postprocessing with audited tolerance for missing decoded tail frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dlc.hybrid import _hybrid_pipeline_impl as pipeline
from dlc.hybrid.dlc_runtime import map_pose_table_to_source_tolerant


def postprocess_hybrid_predictions(
    predictions_h5: Path, transforms_csv: Path, source_video: Path, roi_json: Path,
    output_dir: Path, width_cm: float, height_cm: float, pcutoff: float,
    max_gap_sec: float, median_window: int, make_overlay: bool, draw_all_keypoints: bool,
) -> Path:
    """Fuse DLC predictions and explicitly audit a small missing video suffix."""
    from dlc.postprocess import (
        _fill_short_gaps, fuse_pose, load_arena_transform, normalize_dlc_dataframe,
        transform_points, write_overlay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_hdf(predictions_h5)
    crop_table = normalize_dlc_dataframe(raw)
    table, transforms, dropped_tail = map_pose_table_to_source_tolerant(crop_table, transforms_csv)
    fused = fuse_pose(table, pcutoff)
    source_frames, fps, _width, _height = pipeline._video_properties(source_video)
    missing_source_tail = source_frames - table.frame_count
    missing_fraction = missing_source_tail / max(1, source_frames)
    if missing_source_tail < 0 or missing_fraction > 0.05:
        raise ValueError(
            "Source/prediction frame mismatch is not a small suffix shortfall: "
            f"{source_frames} source frames vs {table.frame_count} predictions "
            f"({abs(missing_source_tail)} frames, {abs(missing_fraction):.2%})."
        )
    if missing_source_tail > 0 and missing_source_tail != dropped_tail:
        print(
            f"WARNING: source video has {missing_source_tail} unpredicted tail frames; "
            "the exported trajectory ends at the final decoded DLC prediction.",
            flush=True,
        )

    gap = max(0, int(round(max_gap_sec * fps)))
    body_raw, head_raw = fused["body"], fused["head"]
    body = _fill_short_gaps(body_raw, gap, median_window)
    head = _fill_short_gaps(head_raw, gap, median_window)
    homography = load_arena_transform(roi_json, width_cm, height_cm)
    body_cm, head_cm = transform_points(body, homography), transform_points(head, homography)
    n = table.frame_count
    trajectory = pd.DataFrame({
        "frame": np.arange(n), "timestamp_sec": np.arange(n) / fps,
        "body_x_cm": body_cm[:, 0], "body_y_cm": body_cm[:, 1],
        "head_x_cm": head_cm[:, 0], "head_y_cm": head_cm[:, 1],
        "body_x_px": body[:, 0], "body_y_px": body[:, 1],
        "head_x_px": head[:, 0], "head_y_px": head[:, 1],
        "body_x_px_raw": body_raw[:, 0], "body_y_px_raw": body_raw[:, 1],
        "head_x_px_raw": head_raw[:, 0], "head_y_px_raw": head_raw[:, 1],
        "body_confidence": fused["body_confidence"], "head_confidence": fused["head_confidence"],
        "body_interpolated": np.isfinite(body[:, 0]) & ~np.isfinite(body_raw[:, 0]),
        "head_interpolated": np.isfinite(head[:, 0]) & ~np.isfinite(head_raw[:, 0]),
        "yolo_confidence": transforms.detector_confidence,
        "yolo_box_source": transforms.box_source,
    })
    csv_path = output_dir / "trajectory.csv"
    trajectory.to_csv(csv_path, index=False, float_format="%.5f")
    raw.to_hdf(output_dir / "dlc_crop_keypoints.h5", key="df_with_missing", mode="w")
    mapped_columns = {}
    for name, values in table.points.items():
        mapped_columns[(name, "x")] = values[:, 0]
        mapped_columns[(name, "y")] = values[:, 1]
        mapped_columns[(name, "likelihood")] = values[:, 2]
    pd.DataFrame(mapped_columns).to_csv(output_dir / "dlc_source_keypoints.csv", index=False)
    qa = {
        "source_video": str(source_video.resolve()),
        "crop_predictions": str(predictions_h5.resolve()),
        "crop_transforms": str(transforms_csv.resolve()),
        "source_frames": source_frames,
        "prediction_frames": n,
        "omitted_tail_frames": max(0, missing_source_tail),
        "omitted_tail_percent": round(100 * max(0.0, missing_fraction), 3),
        "fps": fps,
        "yolo_direct_percent": round(100 * float((transforms.box_source == "yolo").mean()), 3),
        "yolo_fallback_percent": round(100 * float((transforms.box_source == "temporal_fallback").mean()), 3),
        "yolo_missing_percent": round(100 * float((transforms.box_source == "missing").mean()), 3),
        "body_valid_percent": round(100 * float(np.isfinite(body[:, 0]).mean()), 3),
        "head_valid_percent": round(100 * float(np.isfinite(head[:, 0]).mean()), 3),
        "bodyparts_found": sorted(table.points),
    }
    (output_dir / "quality_report.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if make_overlay:
        write_overlay(source_video, output_dir / "annotated_output.mp4", trajectory, table, pcutoff, draw_all_keypoints)
    return csv_path
