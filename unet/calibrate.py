#!/usr/bin/env python3
"""Arena calibration helpers shared by the U-Net pipeline.

Two problems are solved here:

1. Wall-band exclusion (floor detection). The ROI often includes the arena
   walls because the camera sees the mouse's shadow projected on them. The
   wall content changes with the mouse position and must never influence
   segmentation. `detect_floor_bounds` finds the floor rectangle inside the
   rectified background (the wall band is either brighter or darker than the
   floor, so both hypotheses are tried); the caller then zeroes the mask
   outside the floor.

2. Corner refinement (coordinate calibration). Four clicks on the arena
   corners carry a few pixels of error; different videos then get slightly
   different rectified frames. `refine_corners` snaps the clicks to the
   detected arena edges (Hough lines) and returns the exact intersections,
   so every video shares one coordinate system.
"""
from __future__ import annotations

import numpy as np
import cv2


def detect_floor_bounds(background: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect the floor rectangle inside a rectified background image.

    Walls are a border band that is clearly brighter OR darker than the
    floor; both hypotheses are tried. Returns floor bounds (x0, y0, x1, y1)
    in rectified pixels, or None when the floor cannot be separated (no
    wall contrast) - the caller then treats the whole image as floor.
    """
    g = cv2.GaussianBlur(background, (5, 5), 0)
    h, w = g.shape
    if h < 30 or w < 30:
        return None
    # Otsu splits the bimodal floor/wall brightness distribution; the floor
    # is then either the bright or the dark side (both are tried).
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = []
    for mask in (g > thr, g < thr):
        m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((15, 15), np.uint8))
        n, labels, stats, cent = cv2.connectedComponentsWithStats(m)
        for label in range(1, n):
            x, y, ww, hh, area = stats[label]
            if area < 0.3 * w * h:
                continue  # floor must be the dominant region
            if area < 0.6 * ww * hh:
                continue  # solidity: floor is a filled rectangle, a wall
                # frame or noise blob is sparse inside its bounding box
            cx, cy = cent[label]
            if abs(cx - w / 2) < 0.3 * w and abs(cy - h / 2) < 0.3 * h:
                candidates.append((area, x, y, x + ww, y + hh))
                break
    if not candidates:
        return None
    _, x0, y0, x1, y1 = max(candidates, key=lambda c: c[0])
    if x0 < 2 or y0 < 2 or x1 > w - 2 or y1 > h - 2:
        return None  # floor touching the image edge: no visible wall band
    return (int(x0), int(y0), int(x1), int(y1))


def _line_from_pts(x1, y1, x2, y2) -> tuple[float, float, float]:
    """Line ax+by+c=0 through two points, normalized."""
    a, b = y2 - y1, x1 - x2
    norm = np.hypot(a, b)
    if norm == 0:
        return (0.0, 0.0, 0.0)
    a, b = a / norm, b / norm
    c = -(a * x1 + b * y1)
    return (a, b, c)


def _line_intersect(l1, l2) -> tuple[float, float] | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return ((b1 * c2 - b2 * c1) / det, (a2 * c1 - a1 * c2) / det)


def refine_corners(gray: np.ndarray, approx: np.ndarray,
                   angle_tol: float = 20.0) -> np.ndarray:
    """Snap 4 arena corners to the detected arena edges.

    `approx` is (4, 2) array of clicked corner pixels (any order is fine;
    the quadrilateral edge directions are taken from the clicks). Every
    edge is matched against Hough lines with similar direction located near
    the click; the 4 selected lines intersect at the refined corners.
    Edges without a reliable line keep the original click.
    """
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=max(50, min(g.shape) // 5),
                            maxLineGap=10)
    if lines is None or len(lines) < 4:
        return approx.copy().astype(float)
    pts = np.asarray(lines).reshape(-1, 4).astype(float)  # (N,4) x1,y1,x2,y2
    # quadrilateral edges: approx[i] -> approx[(i+1)%4]
    best_lines: list[tuple[float, float, float] | None] = [None] * 4
    for i in range(4):
        p0, p1 = approx[i], approx[(i + 1) % 4]
        ex, ey = p1 - p0
        e_angle = np.degrees(np.arctan2(ey, ex))
        edge_mid = (p0 + p1) / 2
        best_dist, best_line = None, None
        for (x1, y1, x2, y2) in pts:
            l_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            diff = abs((l_angle - e_angle) % 180)   # line angle is undirected
            if diff > 90:
                diff = 180 - diff
            if diff > angle_tol:
                continue
            a, b, c = _line_from_pts(x1, y1, x2, y2)
            dist = abs(a * edge_mid[0] + b * edge_mid[1] + c)  # to edge midpoint
            if best_dist is None or dist < best_dist:
                best_dist, best_line = dist, (a, b, c)
        best_lines[i] = best_line
    refined = approx.copy().astype(float)
    for i in range(4):
        l_prev, l_next = best_lines[(i - 1) % 4], best_lines[i]
        if l_prev is None or l_next is None:
            continue
        ip = _line_intersect(l_prev, l_next)
        if ip is None:
            continue
        x, y = ip
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            refined[i] = (x, y)
    return refined