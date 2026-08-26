#!/usr/bin/env python3
import unittest

import pandas as pd

from gui.incremental_review import opposite_head_frames, spaced_worst


class IncrementalReviewTests(unittest.TestCase):
    def test_only_opposite_centroid_sides_are_selected(self):
        data = pd.DataFrame({
            "frame": [10, 20],
            "body_x_cm": [5, 5], "body_y_cm": [5, 5],
            "head_x_cm": [7, 7], "head_y_cm": [5, 5],
            "reflection_x_cm": [3, 6], "reflection_y_cm": [5, 5],
        })
        self.assertEqual([frame for frame, _ in opposite_head_frames(data)], [10])

    def test_near_duplicate_candidates_are_spaced(self):
        rows = [(10, .1), (11, .2), (20, .3)]
        self.assertEqual([frame for frame, _ in spaced_worst(rows, 10, 4)],
                         [10, 20])


if __name__ == "__main__":
    unittest.main()
