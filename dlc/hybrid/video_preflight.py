"""Preflight an existing prepared crop video before expensive DLC inference."""

from __future__ import annotations

import json
from pathlib import Path

from dlc.hybrid.stable_video import _decoded_frame_count


def validate_prepared_video(cfg: dict, source_video: Path) -> None:
    import pandas as pd

    folder = Path(cfg["output_dir"]) / "hybrid" / source_video.stem
    manifest_path = folder / "hybrid_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare the YOLO/SR input first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video = Path(manifest["dlc_input_video"])
    transforms = Path(manifest["transforms"])
    expected = len(pd.read_csv(transforms, usecols=["frame"]))
    decoded = _decoded_frame_count(video)
    if decoded != expected:
        raise RuntimeError(
            "Prepared DLC input is not safe to analyze: "
            f"crop_transforms.csv contains {expected} frames but {video.name} fully decodes "
            f"only {decoded}. Run '1 · YOLO + SR preparation' again with the updated code; "
            "it will create and verify a stable MJPEG/AVI intermediate."
        )
    print(f"DLC input preflight passed: {decoded}/{expected} decodable frames", flush=True)
