"""Runtime validation and frame-alignment policy for hybrid DLC jobs."""

from __future__ import annotations

from pathlib import Path


def validate_video_adaptation(cfg: dict) -> None:
    """Reject an adaptation setup that DLC can only fail after inference."""
    model = cfg.get("model", {})
    if not bool(model.get("video_adapt", True)):
        return
    detector_epochs = int(model.get("detector_epochs", 1))
    pose_epochs = int(model.get("pose_epochs", 1))
    if detector_epochs < 1 or pose_epochs < 1:
        raise ValueError(
            "Self-supervised video adaptation is enabled, but detector_epochs="
            f"{detector_epochs} and pose_epochs={pose_epochs}. DeepLabCut needs both "
            "values to be at least 1 to create the adapted checkpoints. Either disable "
            "'Self-supervised video adaptation' for normal model-zoo inference, or set "
            "both adaptation epoch values to 1 or higher."
        )


def map_pose_table_to_source_tolerant(
    table, transforms_csv: Path, max_tail_fraction: float = 0.05
):
    """Map predictions while tolerating a small, suffix-only decoder shortfall."""
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    transforms = pd.read_csv(transforms_csv)
    prediction_count = int(table.frame_count)
    transform_count = len(transforms)
    dropped_tail = transform_count - prediction_count
    if dropped_tail:
        fraction = dropped_tail / max(1, transform_count)
        if dropped_tail < 0 or fraction > max_tail_fraction:
            raise ValueError(
                "Transform/prediction frame mismatch is not a small suffix shortfall: "
                f"{transform_count} transforms vs {prediction_count} predictions "
                f"({abs(dropped_tail)} frames, {abs(fraction):.2%})."
            )
        print(
            "WARNING: DeepLabCut decoded fewer frames than the prepared video; "
            f"using the aligned first {prediction_count} frames and omitting the final "
            f"{dropped_tail} frames ({fraction:.2%}).",
            flush=True,
        )
        transforms = transforms.iloc[:prediction_count].reset_index(drop=True)

    x0 = pd.to_numeric(transforms.x0, errors="coerce").to_numpy(float)
    y0 = pd.to_numeric(transforms.y0, errors="coerce").to_numpy(float)
    size = pd.to_numeric(transforms.crop_size, errors="coerce").to_numpy(float)
    output_size = pd.to_numeric(transforms.output_size, errors="coerce").to_numpy(float)
    mapped = {}
    for name, values in table.points.items():
        values = values.copy()
        valid = (
            np.isfinite(values[:, :2]).all(axis=1)
            & np.isfinite(x0 + y0 + size + output_size)
            & (output_size > 0)
        )
        values[valid, 0] = x0[valid] + values[valid, 0] * size[valid] / output_size[valid]
        values[valid, 1] = y0[valid] + values[valid, 1] * size[valid] / output_size[valid]
        values[~valid, :2] = np.nan
        mapped[name] = values
    return replace(table, points=mapped), transforms, max(0, dropped_tail)
