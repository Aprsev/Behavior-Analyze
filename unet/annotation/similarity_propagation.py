"""Conservative propagation from a manual anchor to near-identical video frames."""
from __future__ import annotations

import cv2
import numpy as np


def appearance_descriptor(frame: np.ndarray, size: int = 96) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height, width = gray.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(width * scale), round(height * scale)),
                          interpolation=cv2.INTER_AREA)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray -= float(gray.mean())
    gray /= max(float(gray.std()), 1.0)
    return gray


def frame_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Normalized appearance correlation mapped to [0, 1]."""
    a, b = appearance_descriptor(first), appearance_descriptor(second)
    correlation = float(np.mean(a * b))
    return float(np.clip((correlation + 1.0) * .5, 0.0, 1.0))


def read_frame(cap: cv2.VideoCapture, frame: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = cap.read()
    return image if ok else None


def similar_candidate_neighbors(
        cap: cv2.VideoCapture, anchor_frame: int, candidates: set[int],
        threshold: float = .97, window: int = 15) -> list[tuple[int, float]]:
    """Return same-looking candidates around an anchor, stopping on change.

    Search proceeds independently forward/backward and stops as soon as a
    frame falls below the threshold. This prevents propagation across a real
    posture transition even when a later frame happens to look similar again.
    """
    anchor = read_frame(cap, anchor_frame)
    if anchor is None:
        return []
    found: list[tuple[int, float]] = []
    for direction in (-1, 1):
        for offset in range(1, max(0, int(window)) + 1):
            frame = anchor_frame + direction * offset
            if frame < 0:
                break
            image = read_frame(cap, frame)
            if image is None:
                break
            similarity = frame_similarity(anchor, image)
            if similarity < threshold:
                break
            if frame in candidates:
                found.append((frame, similarity))
    return sorted(found)


def translate_polygon_optical_flow(
        source: np.ndarray, target: np.ndarray, polygon: np.ndarray,
        max_shift_px: float = 12.0) -> np.ndarray | None:
    """Translate a sparse polygon by robust median LK flow.

    Individual vertices often lie on textureless fur/background boundaries;
    the median valid displacement is intentionally used instead of deforming
    each vertex independently. High similarity plus a small median shift is a
    safer constraint for stationary-mouse runs.
    """
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    points = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        source_gray, target_gray, points, None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    if moved is None or status is None:
        return None
    valid = status.reshape(-1) > 0
    if int(valid.sum()) < max(3, int(np.ceil(len(points) * .5))):
        return None
    displacement = np.median(
        moved.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid], axis=0)
    if not np.isfinite(displacement).all() or np.linalg.norm(displacement) > max_shift_px:
        return None
    result = points.reshape(-1, 2) + displacement
    height, width = source_gray.shape
    result[:, 0] = np.clip(result[:, 0], 0, width - 1)
    result[:, 1] = np.clip(result[:, 1], 0, height - 1)
    return result
