"""Anatomical and temporal constraints for tethered-mouse head candidates."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from head_fusion import HeadChoice


@dataclass
class AnatomyResult:
    choice: HeadChoice
    elongation: float
    tail_detected: bool
    corrected: bool
    outside_distance_px: float


def _main_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour[:, 0].astype(np.float32) if len(contour) >= 5 else None


def _nearest_contour(point: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, float]:
    distance = np.sqrt(np.sum((contour - point[None]) ** 2, axis=1))
    index = int(distance.argmin())
    return contour[index].copy(), float(distance[index])


def _principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Analytic 2-D PCA without BLAS/LAPACK/OpenMP runtime initialization."""
    points = np.asarray(points, np.float32).reshape(-1, 2)
    center = points.mean(axis=0)
    delta = points - center
    sxx = float(np.mean(delta[:, 0] * delta[:, 0]))
    syy = float(np.mean(delta[:, 1] * delta[:, 1]))
    sxy = float(np.mean(delta[:, 0] * delta[:, 1]))
    discriminant = math.sqrt(max(0.0, (sxx - syy) ** 2 + 4.0 * sxy * sxy))
    major = 0.5 * (sxx + syy + discriminant)
    minor = 0.5 * (sxx + syy - discriminant)
    if abs(sxy) > 1e-12:
        axis = np.asarray([major - syy, sxy], np.float32)
    elif sxx >= syy:
        axis = np.asarray([1.0, 0.0], np.float32)
    else:
        axis = np.asarray([0.0, 1.0], np.float32)
    axis /= max(math.hypot(float(axis[0]), float(axis[1])), 1e-6)
    return center, axis, major, max(minor, 0.0)


def clamp_choice_to_mask(choice: HeadChoice, mask: np.ndarray) -> HeadChoice:
    """Guarantee an automatic result is on the current clean animal mask."""
    if choice.point is None:
        return choice
    point = np.asarray(choice.point, float)
    if not np.isfinite(point).all():
        return HeadChoice(None, 0.0, "missing", choice.disagreement_px)
    x, y = np.rint(point).astype(int)
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
        return choice
    contour = _main_contour(mask)
    if contour is None:
        return HeadChoice(None, 0.0, "anatomy_no_mask", choice.disagreement_px)
    nearest, _ = _nearest_contour(point, contour)
    return HeadChoice(tuple(map(float, nearest)), choice.confidence * 0.75,
                      "mask_clamped_" + choice.source, choice.disagreement_px)


class AnatomicalHeadConstraint:
    """Resolve head/tail ambiguity using shape, appendages and continuity."""

    def __init__(self, elongated_ratio: float = 1.35, flip_confirm_frames: int = 4):
        self.elongated_ratio = float(elongated_ratio)
        self.flip_confirm_frames = max(2, int(flip_confirm_frames))
        self.previous_direction: np.ndarray | None = None
        self.flip_votes = 0

    def reset(self) -> None:
        self.previous_direction = None
        self.flip_votes = 0

    @staticmethod
    def geometry(mask: np.ndarray):
        contour = _main_contour(mask)
        if contour is None:
            return None
        center, axis, major, minor = _principal_axis(contour)
        elongation = float(math.sqrt(max(major, 1e-6) / max(minor, 1e-6)))
        projection = (contour - center) @ axis
        low_cut, high_cut = np.percentile(projection, [4, 96])
        low = contour[projection <= low_cut].mean(axis=0)
        high = contour[projection >= high_cut].mean(axis=0)
        major_length = float(projection.max() - projection.min())
        return contour, center, (low, high), elongation, major_length

    @staticmethod
    def tail_endpoint(body_mask: np.ndarray, foreground: np.ndarray,
                      endpoints: tuple[np.ndarray, np.ndarray]) -> int | None:
        """Return which major-axis endpoint is connected to a contained tail."""
        body = (body_mask > 0).astype(np.uint8)
        fg = (foreground > 0).astype(np.uint8)
        if not body.any() or not fg.any():
            return None
        inner = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        shell = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        extension = ((fg > 0) & (inner == 0)).astype(np.uint8)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(extension, 8)
        height, width = extension.shape
        body_points = np.column_stack(np.where(body > 0))[:, ::-1]
        body_span = float(np.ptp(body_points, axis=0).max()) if len(body_points) else 1.0
        choices = []
        for label in range(1, n):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not 4 <= area <= max(500, int(body.sum() * 0.65)):
                continue
            if not np.any(component & (shell > 0)):
                continue
            x, y, w, h = (int(stats[label, key]) for key in (
                cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            component_points = np.column_stack(np.where(component))[:, ::-1].astype(np.float32)
            if len(component_points) >= 3:
                _, _, branch_major, branch_minor = _principal_axis(component_points)
                slenderness = float(math.sqrt(max(branch_major, 1e-6) /
                                              max(branch_minor, 1e-6)))
            else:
                slenderness = 1.0
            length = float(max(w, h))
            if length < max(7.0, 0.16 * body_span) or slenderness < 2.0:
                continue
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                continue
            center = centers[label].astype(np.float32)
            endpoint_index = int(np.argmin([
                math.hypot(float(center[0] - point[0]), float(center[1] - point[1]))
                for point in endpoints]))
            score = length * min(slenderness, 8.0) + 0.05 * area
            choices.append((score, endpoint_index))
        return max(choices, default=(0.0, None), key=lambda item: item[0])[1]

    def update(self, body, body_mask: np.ndarray, foreground: np.ndarray,
               choice: HeadChoice) -> AnatomyResult:
        geometry = self.geometry(body_mask)
        if body is None or geometry is None:
            return AnatomyResult(choice, 1.0, False, False, 0.0)
        contour, _, endpoints, elongation, major_length = geometry
        b = np.asarray(body, float)
        candidate = np.asarray(choice.point, float) if choice.point is not None else None
        corrected = False; outside_distance = 0.0
        if candidate is not None and np.isfinite(candidate).all():
            x, y = np.rint(candidate).astype(int)
            inside = (0 <= y < body_mask.shape[0] and 0 <= x < body_mask.shape[1]
                      and body_mask[y, x] > 0)
            if not inside:
                candidate, outside_distance = _nearest_contour(candidate, contour)
                corrected = True
        else:
            candidate = None

        tail_index = self.tail_endpoint(body_mask, foreground, endpoints)
        tail_detected = tail_index is not None
        directions = []
        for endpoint in endpoints:
            direction = endpoint - b
            directions.append(direction / max(
                math.hypot(float(direction[0]), float(direction[1])), 1e-6))

        candidate_index = None
        reflection_direction_cue = False
        if candidate is not None:
            candidate_index = int(np.argmin([
                math.hypot(float(candidate[0] - point[0]), float(candidate[1] - point[1]))
                for point in endpoints]))
            candidate_axial_distance = abs(float(np.dot(
                candidate - b, directions[candidate_index])))
            reflection_direction_cue = (
                "reflection" in choice.source and outside_distance <= 3.0 and
                candidate_axial_distance >= 0.12 * major_length)

        tail_head_index = 1 - tail_index if tail_index is not None else None
        reflection_tail_consensus = (tail_head_index is not None and
                                     reflection_direction_cue and
                                     candidate_index == tail_head_index)
        if tail_head_index is not None:
            # A contained tail disagrees with a Reflection on the same side:
            # treat that spot as a tail/fibre false positive.
            head_index = tail_head_index
        elif reflection_direction_cue:
            head_index = candidate_index
        elif self.previous_direction is not None:
            head_index = int(np.argmax([float(np.dot(direction, self.previous_direction))
                                        for direction in directions]))
        elif candidate_index is not None:
            head_index = candidate_index
        else:
            return AnatomyResult(choice, elongation, False, corrected, outside_distance)

        proposed_direction = directions[head_index]
        if self.previous_direction is not None and np.dot(proposed_direction, self.previous_direction) < 0:
            self.flip_votes += 1
            required_votes = (1 if reflection_tail_consensus else
                              2 if reflection_direction_cue else self.flip_confirm_frames)
            if self.flip_votes < required_votes:
                head_index = int(np.argmax([float(np.dot(direction, self.previous_direction))
                                            for direction in directions]))
                proposed_direction = directions[head_index]
        else:
            self.flip_votes = 0

        head_endpoint = endpoints[head_index]
        if elongation >= self.elongated_ratio:
            wrong_half = (candidate is not None and
                          float(np.dot(candidate - b, proposed_direction)) < -0.05 * major_length)
            far_outside = outside_distance > max(3.0, 0.12 * major_length)
            if candidate is None or wrong_half or far_outside:
                candidate = head_endpoint.copy()
                corrected = True
                source = "anatomical_endpoint_" + choice.source
                confidence = min(0.65, max(0.12, choice.confidence * 0.75))
            else:
                source = ("anatomical_clamped_" if outside_distance else "anatomical_") + choice.source
                confidence = choice.confidence * (0.75 if outside_distance else 1.0)
        else:
            if candidate is None:
                return AnatomyResult(choice, elongation, tail_detected, corrected, outside_distance)
            source = ("anatomical_clamped_" if outside_distance else "anatomical_") + choice.source
            confidence = choice.confidence * (0.75 if outside_distance else 1.0)

        self.previous_direction = proposed_direction if self.previous_direction is None else (
            0.82 * self.previous_direction + 0.18 * proposed_direction)
        self.previous_direction /= max(
            math.hypot(float(self.previous_direction[0]), float(self.previous_direction[1])), 1e-6)
        constrained = HeadChoice(tuple(map(float, candidate)), float(confidence), source,
                                 choice.disagreement_px)
        constrained = clamp_choice_to_mask(constrained, body_mask)
        return AnatomyResult(constrained, elongation, tail_detected, corrected,
                             outside_distance)


def appendage_foreground(rectified_frame: np.ndarray, background: np.ndarray,
                         body_mask: np.ndarray,
                         raw_model_mask: np.ndarray | None = None) -> np.ndarray:
    """Thin structures near the body, before opening removes tail/fibre."""
    gray = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    dark = cv2.subtract(bg_gray, gray)
    local = cv2.dilate((body_mask > 0).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
    values = dark[local]
    if not values.size:
        return np.zeros_like(body_mask)
    threshold = max(12.0, float(np.percentile(values, 68)))
    foreground = ((dark >= threshold) & local).astype(np.uint8) * 255
    if raw_model_mask is not None and raw_model_mask.shape == foreground.shape:
        # Preserve thin U-Net foreground that the configured opening removed;
        # static-background contrast supplies a second, independent cue.
        foreground[(raw_model_mask > 0) & local] = 255
    return cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
