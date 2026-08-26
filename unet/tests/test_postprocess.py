"""Synthetic regression tests for fibre-aware temporal mask filtering."""
from __future__ import annotations

import unittest
import cv2
import numpy as np

from core.postprocess import TemporalMaskFilter


def mouse_with_fibre(center=(80, 150), fibre_end=(230, 20)) -> np.ndarray:
    probability = np.zeros((256, 256), np.float32)
    cv2.ellipse(probability, center, (19, 12), 0, 0, 360, .94, -1)
    cv2.circle(probability, (center[0] + 14, center[1]), 8, .96, -1)
    cv2.line(probability, (center[0] + 16, center[1] - 3), fibre_end, .92, 2)
    return probability


class FibreFilterTests(unittest.TestCase):
    def test_thin_fibre_does_not_pull_centroid(self):
        tracker = TemporalMaskFilter(opening_px=5)
        result = tracker.update(mouse_with_fibre(), .5)
        self.assertIsNotNone(result.centroid)
        self.assertLess(np.hypot(result.centroid[0] - 83, result.centroid[1] - 150), 5)
        self.assertLess(np.count_nonzero(result.mask[:, 180:]), 20)

    def test_moving_fibre_does_not_move_stationary_mouse(self):
        tracker = TemporalMaskFilter(opening_px=5)
        centers = []
        for endpoint in ((230, 20), (240, 100), (220, 210), (150, 10)):
            result = tracker.update(mouse_with_fibre(fibre_end=endpoint), .5)
            centers.append(result.centroid)
        self.assertTrue(all(c is not None for c in centers))
        self.assertLess(max(c[0] for c in centers) - min(c[0] for c in centers), 2)
        self.assertLess(max(c[1] for c in centers) - min(c[1] for c in centers), 2)

    def test_reacquires_after_external_relocation(self):
        tracker = TemporalMaskFilter(opening_px=5, hold_frames=2, reacquire_frames=5)
        tracker.update(mouse_with_fibre(center=(60, 190)), .5)
        result = None
        for _ in range(8):
            result = tracker.update(mouse_with_fibre(center=(190, 55), fibre_end=(40, 10)), .5)
        self.assertIsNotNone(result.centroid)
        self.assertLess(np.hypot(result.centroid[0] - 193, result.centroid[1] - 55), 6)


if __name__ == "__main__":
    unittest.main()
