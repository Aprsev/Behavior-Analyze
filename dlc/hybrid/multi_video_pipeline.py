"""Training-video pseudo labels and leakage-safe YOLO dataset export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import cv2
import numpy as np
import pandas as pd

from dlc.hybrid import _hybrid_pipeline_impl as original
from dlc.hybrid.dataset_config import config_for_record, records


def generate_training_boxes(cfg: dict[str, Any]) -> Path:
    """Generate pseudo labels from training videos only, with per-video ROIs."""
    items = records(cfg, "train")
    if not items:
        raise ValueError("No training videos configured")
    root = Path(cfg["yolo"]["dataset_dir"]).resolve()
    images = root / "source_images"
    images.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="hybrid_boxes_") as temporary:
        temporary_root = Path(temporary)
        for index, item in enumerate(items):
            if not Path(item["video"]).is_file():
                raise FileNotFoundError(f"Training video is missing: {item['video']}")
            if not Path(item["roi_json"]).is_file():
                raise FileNotFoundError(f"Training ROI is missing: {item['roi_json']}")
            part_cfg = config_for_record(cfg, item)
            part_dataset = temporary_root / f"part_{index:04d}"
            part_cfg["yolo"]["dataset_dir"] = str(part_dataset)
            labels_path = original.generate_traditional_boxes(part_cfg)
            frame = pd.read_csv(labels_path)
            for row_index, row in frame.iterrows():
                source = Path(str(row.image))
                token = hashlib.sha256(str(Path(item["video"]).resolve()).encode()).hexdigest()[:10]
                destination = images / f"{token}_{source.name}"
                shutil.copy2(source, destination)
                frame.at[row_index, "image"] = str(destination.resolve())
            frame["dataset_split"] = "train_source"
            all_rows.append(frame)
    combined = pd.concat(all_rows, ignore_index=True)
    labels = root / "box_labels.csv"
    combined.to_csv(labels, index=False, float_format="%.5f")
    print(f"Wrote {len(combined)} training-only pseudo-labels from {len(items)} videos to {labels}", flush=True)
    return labels


def _video_validation_set(videos: list[str], fraction: float, seed: int) -> set[str]:
    unique = sorted(set(videos))
    if len(unique) < 2:
        return set()
    count = min(len(unique) - 1, max(1, int(round(len(unique) * fraction))))
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    return set(ranked[:count])


def export_grouped_yolo_dataset(cfg: dict[str, Any]) -> Path:
    """Export train/val with whole-video grouping to prevent adjacent-frame leakage."""
    ycfg = cfg.get("yolo", {})
    root = Path(ycfg["dataset_dir"]).resolve()
    rows = original.load_box_labels(root / "box_labels.csv")
    valid = rows.loc[
        ~rows.exclude & rows[["x1", "y1", "x2", "y2"]].notna().all(axis=1)
    ].copy()
    if len(valid) < 2:
        raise ValueError("At least two non-excluded boxes are required")
    fraction = float(ycfg.get("validation_fraction", 0.2))
    seed = int(ycfg.get("split_seed", 42))
    held_out = _video_validation_set(valid.video.astype(str).tolist(), fraction, seed)
    if held_out:
        valid["split"] = np.where(valid.video.astype(str).isin(held_out), "val", "train")
        split_mode = "whole_video"
    else:
        assignments = []
        for _, row in valid.iterrows():
            token = f"{seed}:{row.video}:{int(row.frame)}".encode()
            value = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
            assignments.append("val" if value < fraction else "train")
        valid["split"] = assignments
        if not (valid.split == "val").any():
            valid.iloc[-1, valid.columns.get_loc("split")] = "val"
        if not (valid.split == "train").any():
            valid.iloc[0, valid.columns.get_loc("split")] = "train"
        split_mode = "frame_fallback_single_video"

    for kind in ("images", "labels"):
        for split in ("train", "val"):
            folder = root / kind / split
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
    written = {"train": 0, "val": 0}
    for _, row in valid.iterrows():
        source = Path(str(row.image))
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Cannot read dataset image: {source}")
        height, width = image.shape[:2]
        x1, y1 = np.clip([float(row.x1), float(row.y1)], [0, 0], [width - 1, height - 1])
        x2, y2 = np.clip([float(row.x2), float(row.y2)], [0, 0], [width - 1, height - 1])
        if x2 <= x1 or y2 <= y1:
            continue
        split = str(row.split)
        destination = root / "images" / split / source.name
        shutil.copy2(source, destination)
        xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        (root / "labels" / split / f"{source.stem}.txt").write_text(
            f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8"
        )
        written[split] += 1
    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: mouse\n",
        encoding="utf-8",
    )
    manifest = {
        "labels_csv": str((root / "box_labels.csv").resolve()),
        "split_mode": split_mode,
        "validation_videos": sorted(held_out),
        "external_test_videos": [item["video"] for item in records(cfg, "test")],
        "train_count": written["train"],
        "val_count": written["val"],
        "class": "mouse",
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"YOLO dataset: {manifest}", flush=True)
    return yaml
