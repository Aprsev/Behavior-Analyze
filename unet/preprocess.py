#!/usr/bin/env python3
"""Per-video background estimation + background-centered normalization.

One recording has an almost static background, but across recordings the
arena / lamp / camera can differ a lot. A CNN that ingests raw frames spends
capacity memorizing background appearance; a video that looks different from
every training video then segments badly.

The preprocessing here removes exactly the static part: per video we sample
~61 frames spread over the whole recording and take the per-pixel 85th
percentile (the mouse covers each pixel < 15% of the time, so the percentile
approximates the empty arena). The input to the U-Net is then

    x' = clip(128 + 2 * (gray - bg), 0, 255)

i.e. the static background becomes mid-gray and only deviations from it
(mouse, miniscope, shadow) survive. Training and inference must agree on
this transform, so:

  - prepare_dataset.py  caches the background per video as
    <dataset>/backgrounds/<video_stem>.png  (train.py reads the cache);
  - infer.py / head_track.py compute the background on the fly with the
    very same function, and only apply the transform when the model
    checkpoint says it was trained with it ("bg_subtract": true), so old
    checkpoints keep working untouched.

The transform is applied BEFORE the existing augmentations in train.py
(rotation / flip / warp / lighting / noise all run on the centered image),
so nothing else in the pipeline changes.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_background(video, n: int = 61, percentile: float = 85.0) -> np.ndarray | None:
    """Per-pixel percentile over n frames spread over the video.

    Returns a BGR uint8 background, or None when the video cannot be read /
    has too few decodable frames (callers then fall back to raw input).
    Identical sampling as traditional/code/mouse_behavior_pipeline.sample_frames
    so training-cache and inference-time backgrounds always match.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.unique(np.linspace(0, max(total - 1, 0), min(n, total), dtype=int))
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if len(frames) < 3:
        return None
    return np.percentile(np.stack(frames), percentile, axis=0).astype(np.uint8)


def bg_centered(gray: np.ndarray, bg: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """Static background -> mid-gray; deviations amplified by `gain`.

    Both inputs are uint8 and must have the same shape. Output is uint8 in
    [0, 255]; 128 means "exactly background".
    """
    return np.clip(128.0 + gain * (gray.astype(np.float32) - bg.astype(np.float32)),
                   0, 255).astype(np.uint8)


def save_background(path, bg: np.ndarray) -> None:
    cv2.imwrite(str(path), bg)


def load_background(path) -> np.ndarray | None:
    bg = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return bg if bg is not None else None
