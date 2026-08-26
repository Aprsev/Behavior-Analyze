"""Version-independent automatic frame extraction for supervised DLC labeling."""

from __future__ import annotations

import math
from pathlib import Path


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def _select_kmeans(indices, features, count: int, seed: int) -> list[int]:
    import cv2
    import numpy as np

    if len(indices) <= count:
        return list(indices)
    data = np.asarray(features, dtype=np.float32)
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _compactness, labels, centers = cv2.kmeans(
        data, count, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(-1)
    selected = []
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            continue
        distances = ((data[members] - centers[cluster]) ** 2).sum(axis=1)
        selected.append(indices[int(members[int(distances.argmin())])])
    if len(selected) < count:
        extras = [value for value in _evenly_spaced(indices, count) if value not in selected]
        selected.extend(extras[: count - len(selected)])
    return sorted(selected)


def _scan_candidates(video: Path, algorithm: str, step: int, resize_width: int, color: bool):
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open training video for frame extraction: {video}")
    metadata_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    effective_step = max(1, int(step), math.ceil(metadata_count / 5000) if metadata_count else 1)
    indices, features = [], []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % effective_step == 0:
            indices.append(frame_index)
            if algorithm == "kmeans":
                height = max(1, round(frame.shape[0] * resize_width / frame.shape[1]))
                sample = cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA)
                if not color:
                    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                features.append(sample.reshape(-1))
        frame_index += 1
        if frame_index % 2000 == 0:
            print(f"Scanned {frame_index} frames from {video.name}", flush=True)
    capture.release()
    if frame_index == 0 or not indices:
        raise RuntimeError(f"No frames could be decoded from training video: {video}")
    print(
        f"Candidate scan complete for {video.name}: {frame_index} decoded frames, "
        f"{len(indices)} candidates (step={effective_step})",
        flush=True,
    )
    return indices, features, frame_index


def _write_selected(video: Path, output: Path, selected: list[int], frame_count: int) -> int:
    import cv2

    output.mkdir(parents=True, exist_ok=True)
    targets = set(selected)
    digits = max(1, len(str(max(0, frame_count - 1))))
    capture = cv2.VideoCapture(str(video))
    written = 0
    frame_index = 0
    while targets:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in targets:
            destination = output / f"img{frame_index:0{digits}d}.png"
            if not destination.is_file() and not cv2.imwrite(str(destination), frame):
                capture.release()
                raise RuntimeError(f"Failed to write extracted frame: {destination}")
            targets.remove(frame_index)
            written += 1
        frame_index += 1
    capture.release()
    if targets:
        raise RuntimeError(f"Failed to revisit selected frames in {video.name}: {sorted(targets)}")
    return written


def extract_training_frames(hybrid_cfg: dict, project_config: Path) -> None:
    dataset = hybrid_cfg.get("dataset", {})
    project = hybrid_cfg.get("project", {})
    algorithm = str(dataset.get("algorithm", "kmeans")).lower()
    if str(dataset.get("mode", "automatic")).lower() != "automatic":
        raise ValueError("The integrated extractor currently requires automatic extraction mode")
    if bool(dataset.get("crop", False)):
        raise ValueError("Interactive extraction crop is not supported; crop videos before adding them")
    if algorithm not in {"uniform", "kmeans"}:
        raise ValueError("Frame extraction algorithm must be 'uniform' or 'kmeans'")
    count = int(project.get("num_frames", 40))
    step = int(dataset.get("cluster_step", 1))
    resize_width = int(dataset.get("cluster_resize_width", 30))
    color = bool(dataset.get("cluster_color", False))
    seed = int(hybrid_cfg.get("yolo", {}).get("split_seed", 42))
    videos = [Path(value).expanduser().resolve() for value in hybrid_cfg.get("videos", [])]
    data_root = project_config.resolve().parent / "labeled-data"

    for video_index, video in enumerate(videos):
        print(f"--- extracting DLC frames {video_index + 1}/{len(videos)}: {video.name} ---", flush=True)
        indices, features, frame_count = _scan_candidates(
            video, algorithm, step, resize_width, color
        )
        selected = (
            _select_kmeans(indices, features, count, seed + video_index)
            if algorithm == "kmeans"
            else _evenly_spaced(indices, count)
        )
        written = _write_selected(video, data_root / video.stem, selected, frame_count)
        print(f"Extracted {written} frames to {data_root / video.stem}", flush=True)
