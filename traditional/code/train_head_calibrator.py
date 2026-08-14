#!/usr/bin/env python3
"""Fit a lightweight, interpretable head-position calibrator from manual labels.

The model uses the silhouette head, reflection point, body centre, reflection
confidence, and their body-relative geometry. Reflection and anatomical head
are deliberately separate input signals. A robust ridge regressor is selected
by grouped time-block validation; it can fall back to the silhouette estimate
when the reflection is absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


FEATURES = [
    "body_x_cm", "body_y_cm", "head_silhouette_usable_x_cm", "head_silhouette_usable_y_cm",
    "head_reflection_x_cm", "head_reflection_y_cm", "reflection_confidence",
    "silhouette_usable_dx", "silhouette_usable_dy", "reflection_dx", "reflection_dy",
    "head_reflection_distance_cm", "reflection_present_auto", "silhouette_reliable",
]


def feature_table(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    raw_silhouette_dx = result.head_silhouette_x_cm - result.body_x_cm
    raw_silhouette_dy = result.head_silhouette_y_cm - result.body_y_cm
    result["reflection_dx"] = result.head_reflection_x_cm - result.body_x_cm
    result["reflection_dy"] = result.head_reflection_y_cm - result.body_y_cm
    result["head_reflection_distance_cm"] = np.hypot(result.head_silhouette_x_cm - result.head_reflection_x_cm, result.head_silhouette_y_cm - result.head_reflection_y_cm)
    result["reflection_present_auto"] = result[["head_reflection_x_cm", "head_reflection_y_cm"]].notna().all(axis=1).astype(float)
    # The silhouette endpoint may be the tail or a fibre branch. When a
    # reasonably confident miniscope reflection is present, head and device
    # should be on broadly the same body-centre side. Opposite sides are a
    # strong, geometry-only indication that the red endpoint is a tail/fibre.
    dot = raw_silhouette_dx * result.reflection_dx + raw_silhouette_dy * result.reflection_dy
    norm = np.hypot(raw_silhouette_dx, raw_silhouette_dy) * np.hypot(result.reflection_dx, result.reflection_dy)
    same_side_cosine = dot / norm.replace(0, np.nan)
    reflection_confidence = result.get("reflection_confidence", pd.Series(0.0, index=result.index)).fillna(0.0)
    conflict = (result.reflection_present_auto.astype(bool) & (reflection_confidence >= 0.20) & (same_side_cosine < -0.15))
    result["silhouette_reliable"] = (~conflict & result[["head_silhouette_x_cm", "head_silhouette_y_cm"]].notna().all(axis=1)).astype(float)
    result["head_silhouette_usable_x_cm"] = result.head_silhouette_x_cm.where(result.silhouette_reliable.astype(bool))
    result["head_silhouette_usable_y_cm"] = result.head_silhouette_y_cm.where(result.silhouette_reliable.astype(bool))
    result["silhouette_usable_dx"] = raw_silhouette_dx.where(result.silhouette_reliable.astype(bool))
    result["silhouette_usable_dy"] = raw_silhouette_dy.where(result.silhouette_reliable.astype(bool))
    result["silhouette_conflict_reason"] = np.where(conflict, "opposite_side_of_confident_reflection", "usable_or_no_reflection_anchor")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()
    automatic = pd.read_csv(args.comparison_csv)
    labels = pd.read_csv(args.labels)
    data = feature_table(automatic.merge(labels, on="frame", how="inner", suffixes=("", "_manual")))
    usable = (data.head_present.fillna(False).astype(bool) & ~data.exclude.fillna(False).astype(bool)
              & data[["head_x_cm", "head_y_cm", "body_x_cm", "body_y_cm"]].notna().all(axis=1))
    train = data.loc[usable].copy()
    if len(train) < 20:
        raise ValueError(f"Need at least 20 non-excluded frames with manual anatomical head labels; got {len(train)}")
    # Fill missing automatic candidate coordinates with neutral body-relative
    # values. Presence/confidence features let the regression distinguish this.
    x = train[FEATURES]
    # Learn body-relative head displacement. This is substantially more stable
    # across arena positions than fitting absolute x/y coordinates directly.
    y = train[["head_x_cm", "head_y_cm"]].to_numpy(float) - train[["body_x_cm", "body_y_cm"]].to_numpy(float)
    # Use contiguous time blocks for validation: randomly splitting adjacent
    # frames would exaggerate performance because sequential poses are similar.
    groups = pd.qcut(train.frame.rank(method="first"), q=min(5, max(2, len(train)//20)), labels=False).to_numpy()
    candidates = {
        "ridge": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=1.0)),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.85, random_state=42, n_jobs=-1),
        ),
    }
    validation = {}
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for name, candidate in candidates.items():
        errors = []
        for train_idx, test_idx in splitter.split(x, y, groups):
            candidate.fit(x.iloc[train_idx], y[train_idx])
            pred = candidate.predict(x.iloc[test_idx])
            errors.extend(np.linalg.norm(pred - y[test_idx], axis=1))
        validation[name] = {"median_cm": float(np.median(errors)), "p95_cm": float(np.percentile(errors, 95))}
    selected_name = min(validation, key=lambda name: validation[name]["median_cm"])
    model = candidates[selected_name]
    model.fit(x, y)
    prediction_offset = model.predict(x)
    prediction = prediction_offset + train[["body_x_cm", "body_y_cm"]].to_numpy(float)
    target_xy = train[["head_x_cm", "head_y_cm"]].to_numpy(float)
    error = np.linalg.norm(prediction - target_xy, axis=1)
    # Store plain sklearn model as a pickle: local data only, never load models
    # from untrusted sources.
    import joblib
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "target": "head_offset_from_body_cm", "selected_model": selected_name}, args.model_output)
    metrics = {"training_frames": int(len(train)), "excluded_frames": int((~usable).sum()), "selected_model": selected_name, "training_error_cm_median": round(float(np.median(error)), 4), "training_error_cm_p95": round(float(np.percentile(error, 95)), 4), "time_block_validation_cm": {name: {key: round(value,4) for key,value in score.items()} for name,score in validation.items()}, "silhouette_rejected_as_tail_or_fibre": int((train.silhouette_reliable == 0).sum()), "features": FEATURES, "notes": "Target is head displacement from body. Reflection is a separate predictor and may be absent or offset from anatomical head. Manual head labels are intended as exact overrides during finalization."}
    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
