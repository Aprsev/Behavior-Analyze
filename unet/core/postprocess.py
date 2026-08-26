"""Fibre-aware, temporally stable post-processing for mouse masks.

The tether is a moving dark foreground object, so motion alone cannot
separate it from a stationary mouse.  This module removes thin structures
and scores every remaining component by compactness, network probability and
temporal overlap.  The temporal prior is automatically released after a
short run of misses so an experimenter moving the animal does not permanently
lock the tracker to the old location.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


def _components(mask: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    for label in range(1, n):
        yield label, labels == label, stats[label], centroids[label]


def _compactness(component: np.ndarray) -> float:
    u8 = component.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    area = float(component.sum())
    return float(np.clip(4.0 * np.pi * area / max(perimeter * perimeter, 1.0), 0.0, 1.0))


@dataclass
class MaskResult:
    mask: np.ndarray
    centroid: tuple[float, float] | None
    confidence: float
    status: str


class TemporalMaskFilter:
    """Reject thin tether components while retaining genuine animal motion."""

    def __init__(self, fps: float = 30.0, opening_px: int = 5,
                 hold_frames: int = 3, reacquire_frames: int = 8):
        self.fps = max(float(fps), 1.0)
        self.opening_px = max(3, int(opening_px) | 1)
        self.hold_frames = max(0, int(hold_frames))
        self.reacquire_frames = max(self.hold_frames + 1, int(reacquire_frames))
        self.previous: np.ndarray | None = None
        self.previous_area: float | None = None
        self.misses = 0

    def reset(self) -> None:
        self.previous = None
        self.previous_area = None
        self.misses = 0

    def update(self, probability: np.ndarray, threshold: float = 0.5) -> MaskResult:
        """Return one compact body component from a probability image.

        Work is performed at model resolution, where fibre thickness is
        consistent between source videos.  Opening severs narrow cable
        branches; a light closing restores mouse/miniscope boundary pixels.
        """
        raw = (probability >= float(threshold)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.opening_px, self.opening_px))
        opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
        opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        image_area = opened.size
        previous_dilated = None
        if self.previous is not None and self.misses < self.reacquire_frames:
            previous_dilated = cv2.dilate(
                self.previous, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0

        candidates = []
        for _, component, stats, centroid in _components(opened):
            area = float(stats[cv2.CC_STAT_AREA])
            if not (0.0005 * image_area <= area <= 0.20 * image_area):
                continue
            width = float(stats[cv2.CC_STAT_WIDTH]); height = float(stats[cv2.CC_STAT_HEIGHT])
            aspect = max(width, height) / max(min(width, height), 1.0)
            compact = _compactness(component)
            mean_prob = float(probability[component].mean())
            overlap = float(component[previous_dilated].sum() / max(area, 1.0)) \
                if previous_dilated is not None else 0.0
            area_score = 1.0
            if self.previous_area:
                area_score = float(np.exp(-abs(np.log(max(area, 1.0) / self.previous_area))))
            # Long, narrow components are characteristic of the fibre.  A
            # genuine mouse/miniscope component is compact and normally
            # overlaps a gently expanded previous mask when the mouse rests.
            slender_penalty = max(0.0, aspect - 3.0) * 0.18
            score = 1.5 * mean_prob + 0.9 * compact + 1.2 * overlap + 0.5 * area_score - slender_penalty
            candidates.append((score, component, area, centroid, compact, overlap))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            score, component, area, centroid, compact, overlap = candidates[0]
            # During normal tracking, a completely disjoint thin/highly
            # unstable candidate is treated as a miss.  After several misses
            # the prior is released and the best compact component is accepted
            # at its new location (automatic recovery after manual movement).
            stable = previous_dilated is None or overlap > 0.02 or self.misses >= self.reacquire_frames
            plausible = compact > 0.035
            if stable and plausible:
                result = component.astype(np.uint8) * 255
                self.previous = result
                self.previous_area = area
                self.misses = 0
                return MaskResult(result, (float(centroid[0]), float(centroid[1])),
                                  float(score), "tracked" if overlap else "acquired")

        self.misses += 1
        if self.previous is not None and self.misses <= self.hold_frames:
            m = cv2.moments(self.previous)
            centroid = ((m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else None)
            return MaskResult(self.previous.copy(), centroid, 0.0, "held")
        if self.misses >= self.reacquire_frames:
            self.previous = None
            self.previous_area = None
        return MaskResult(np.zeros_like(raw), None, 0.0, "missing")


def model_input(gray_small: np.ndarray, background_small: np.ndarray | None,
                in_channels: int, gain: float = 2.0) -> np.ndarray:
    """Build the checkpoint-declared input without hiding a stationary mouse."""
    raw = gray_small.astype(np.uint8)
    if in_channels <= 1:
        if background_small is None:
            return raw[None]
        residual = np.clip(128.0 + gain * (raw.astype(np.float32) -
                                           background_small.astype(np.float32)), 0, 255)
        return residual.astype(np.uint8)[None]
    if background_small is None:
        residual = np.full_like(raw, 128)
    else:
        residual = np.clip(128.0 + gain * (raw.astype(np.float32) -
                                           background_small.astype(np.float32)), 0, 255).astype(np.uint8)
    return np.stack([raw, residual], axis=0)
