#!/usr/bin/env python3
import unittest
import numpy as np
import pandas as pd
from head_fusion import choose_head
from annotate_head_results import select_frames


class HeadFusionTests(unittest.TestCase):
    def test_wrong_learned_point_cannot_replace_reflection(self):
        result = choose_head((10, 10), .7, (100, 100), .99)
        self.assertEqual(result.source, "reflection")
        np.testing.assert_allclose(result.point, (10, 10))

    def test_agreeing_learned_point_only_nudges_reflection(self):
        result = choose_head((10, 10), .7, (14, 10), .9)
        self.assertEqual(result.source, "fused_reflection_primary")
        self.assertLess(result.point[0], 11)

    def test_learned_is_used_only_when_reflection_missing(self):
        result = choose_head(None, 0, (20, 30), .8)
        self.assertEqual(result.source, "learned_fallback")
        self.assertEqual(result.point, (20., 30.))

    def test_annotation_selection_prioritizes_failures(self):
        rows = pd.DataFrame({"frame": [1, 2, 3], "head_x_cm": [1, 1, 8],
                             "head_y_cm": [1, 1, 8], "head_confidence": [.9, .2, .8],
                             "head_disagreement_px": [1, 30, 2],
                             "head_source": ["reflection", "learned_fallback", "reflection"]})
        self.assertEqual(select_frames(rows, set(), 1), [2])


if __name__ == "__main__":
    unittest.main()
