"""Create and validate a decoder-stable intermediate video for DeepLabCut."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def _decoded_frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open prepared DLC video: {path}")
    count = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def prepare_hybrid_video_stable(
    original_prepare: Callable[[dict, Path], dict[str, str]], cfg: dict, video: Path
) -> dict[str, str]:
    """Write the crop stream as MJPEG/AVI, then fully decode it before DLC runs."""
    import cv2
    import pandas as pd

    real_writer = cv2.VideoWriter
    stable_path: Path | None = None

    def stable_writer(filename, fourcc, fps, frame_size, *args, **kwargs):
        nonlocal stable_path
        requested = Path(filename)
        if requested.name.lower() == "dlc_input.mp4":
            stable_path = requested.with_suffix(".avi")
            return real_writer(
                str(stable_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, frame_size, *args, **kwargs
            )
        return real_writer(filename, fourcc, fps, frame_size, *args, **kwargs)

    cv2.VideoWriter = stable_writer
    try:
        manifest = original_prepare(cfg, video)
    finally:
        cv2.VideoWriter = real_writer

    if stable_path is None or not stable_path.is_file():
        raise RuntimeError("The stable DLC intermediate video was not created")
    transforms_path = Path(manifest["transforms"])
    expected = len(pd.read_csv(transforms_path, usecols=["frame"]))
    decoded = _decoded_frame_count(stable_path)
    if decoded != expected:
        raise RuntimeError(
            "Prepared DLC video failed the full decode check before inference: "
            f"expected {expected} frames from crop_transforms.csv, decoded {decoded}. "
            "The source recording may be corrupt; re-encode the source video and prepare it again."
        )

    manifest["dlc_input_video"] = str(stable_path.resolve())
    manifest_path = stable_path.parent / "hybrid_manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved["dlc_input_video"] = manifest["dlc_input_video"]
    saved["dlc_input_codec"] = "MJPEG/AVI"
    saved["dlc_input_decoded_frames"] = decoded
    manifest_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"DLC input integrity check passed: {decoded}/{expected} frames ({stable_path.name}, MJPEG/AVI)",
        flush=True,
    )
    return manifest
