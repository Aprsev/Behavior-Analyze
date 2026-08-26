#!/usr/bin/env python3
import unittest

import cv2
import numpy as np

from annotation.similarity_propagation import frame_similarity, translate_polygon_optical_flow


class SimilarityPropagationTests(unittest.TestCase):
    def test_similarity_separates_same_and_changed_posture(self):
        first = np.full((120, 160, 3), 180, np.uint8)
        cv2.ellipse(first, (70, 60), (25, 12), 0, 0, 360, (30, 30, 30), -1)
        same = first.copy()
        changed = np.full_like(first, 180)
        cv2.ellipse(changed, (120, 30), (12, 25), 0, 0, 360, (30, 30, 30), -1)
        self.assertGreater(frame_similarity(first, same), .99)
        self.assertLess(frame_similarity(first, changed), .90)

    def test_polygon_follows_small_translation(self):
        first = np.full((120, 160, 3), 180, np.uint8)
        second = first.copy()
        cv2.rectangle(first, (55, 45), (95, 75), (20, 20, 20), -1)
        cv2.rectangle(second, (58, 47), (98, 77), (20, 20, 20), -1)
        polygon = np.asarray([[55, 45], [95, 45], [95, 75], [55, 75]], np.float32)
        moved = translate_polygon_optical_flow(first, second, polygon)
        self.assertIsNotNone(moved)
        np.testing.assert_allclose(np.median(moved - polygon, axis=0), [3, 2], atol=1.0)


if __name__ == "__main__":
    unittest.main()
