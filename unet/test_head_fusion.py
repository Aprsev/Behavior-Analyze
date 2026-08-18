#!/usr/bin/env python3
import unittest
import numpy as np
import pandas as pd
from head_fusion import HeadChoice, HeadTemporalStabilizer, choose_head, choose_reflection
from annotate_head_results import select_frames


class HeadFusionTests(unittest.TestCase):
    def test_reflection_model_replaces_disagreeing_heuristic_when_confident(self):
        result = choose_reflection((100, 100), .8, (10, 10), .7)
        self.assertEqual(result.source, "reflection_model_disagrees")
        self.assertEqual(result.point, (10., 10.))

    def test_old_checkpoint_still_uses_heuristic_reflection(self):
        result = choose_reflection((10, 10), .6, None, 0)
        self.assertEqual(result.source, "reflection_heuristic")
        self.assertEqual(result.point, (10., 10.))

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

    def test_low_confidence_learned_point_is_not_deleted(self):
        result = choose_head(None, 0, (20, 30), .08)
        self.assertEqual(result.source, "learned_low_confidence_fallback")
        self.assertEqual(result.point, (20., 30.))

    def test_low_confidence_fallback_is_temporally_limited(self):
        stabilizer = HeadTemporalStabilizer(max_gap_frames=5)
        stabilizer.update((10, 10), HeadChoice((15, 10), .8, "reflection", None))
        result = stabilizer.update(
            (11, 10), HeadChoice((-20, 10), .05,
                                 "learned_low_confidence_fallback", None))
        self.assertEqual(result.source, "temporal_learned_low_confidence_fallback")
        self.assertGreater(result.point[0], 10)

    def test_short_missing_gap_uses_body_relative_prediction(self):
        stabilizer = HeadTemporalStabilizer(max_gap_frames=2)
        stabilizer.update((10, 10), HeadChoice((15, 10), .8, "reflection", None))
        result = stabilizer.update((12, 10), HeadChoice(None, 0, "missing", None))
        self.assertEqual(result.source, "temporal_short_gap")
        self.assertEqual(result.point, (17., 10.))

    def test_annotation_selection_prioritizes_failures(self):
        rows = pd.DataFrame({"frame": [1, 2, 3], "head_x_cm": [1, 1, 8],
                             "head_y_cm": [1, 1, 8], "head_confidence": [.9, .2, .8],
                             "head_disagreement_px": [1, 30, 2],
                             "head_source": ["reflection", "learned_fallback", "reflection"]})
        self.assertEqual(select_frames(rows, set(), 1), [2])


if __name__ == "__main__":
    unittest.main()
