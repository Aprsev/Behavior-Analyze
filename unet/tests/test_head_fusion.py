#!/usr/bin/env python3
import unittest
import numpy as np
import pandas as pd
from tracking.head_fusion import HeadChoice, HeadTemporalStabilizer, choose_head, choose_reflection
from annotation.annotate_head_results import display_keypoint_state, select_frames


class HeadFusionTests(unittest.TestCase):
    def test_saved_editor_points_override_automatic_display_but_not_baseline(self):
        automatic = pd.Series({"head_x_cm": 10, "head_y_cm": 5,
                               "reflection_x_cm": 11, "reflection_y_cm": 5})
        saved = pd.Series({"head_x_cm": 3, "head_y_cm": 4,
                           "reflection_x_cm": 2, "reflection_y_cm": 4,
                           "head_present": True, "reflection_present": True,
                           "head_verified": True, "reflection_verified": True})
        state = display_keypoint_state(automatic, saved)
        self.assertEqual(state["head"], (3., 4.))
        self.assertEqual(state["reflection"], (2., 4.))
        self.assertEqual(state["original_head"], (10., 5.))

    def test_saved_absent_point_stays_absent_when_revisited(self):
        automatic = pd.Series({"head_x_cm": 10, "head_y_cm": 5,
                               "reflection_x_cm": 11, "reflection_y_cm": 5})
        saved = pd.Series({"head_x_cm": np.nan, "head_y_cm": np.nan,
                           "reflection_x_cm": 2, "reflection_y_cm": 4,
                           "head_present": False, "reflection_present": True,
                           "head_verified": True, "reflection_verified": True})
        state = display_keypoint_state(automatic, saved)
        self.assertIsNone(state["head"])
        self.assertEqual(state["reflection"], (2., 4.))

    def test_anatomical_reflection_immediately_reseeds_direction(self):
        stabilizer = HeadTemporalStabilizer()
        stabilizer.update((50, 50), HeadChoice((75, 50), .8, "learned_head", None))
        corrected = stabilizer.update(
            (50, 50), HeadChoice((25, 50), .8,
                                 "anatomical_confirmed_reflection", None))
        self.assertLess(corrected.point[0], 50)
        self.assertEqual(corrected.source, "anatomical_confirmed_reflection")

    def test_unconfirmed_reflection_is_smoothed_not_immediately_trusted(self):
        stabilizer = HeadTemporalStabilizer()
        stabilizer.update((50, 50), HeadChoice(
            (75, 50), .8, "anatomical_confirmed_reflection", None))
        result = stabilizer.update((50, 50), HeadChoice(
            (25, 50), .8, "anatomical_reflection_model_disagrees", None))
        self.assertGreater(result.point[0], 50)
        self.assertTrue(result.source.startswith("temporal_"))

    def test_head_choice_preserves_reflection_provenance(self):
        result = choose_head((10, 10), .7, (100, 100), .9,
                             reflection_source="reflection_model_disagrees")
        self.assertIn("model_disagrees", result.source)

    def test_usable_bright_spot_rejects_disagreeing_model(self):
        result = choose_reflection((20, 20), .25, (80, 20), .90)
        self.assertEqual(result.point, (20.0, 20.0))
        self.assertEqual(result.source, "reflection_heuristic_rejects_model")

    def test_weak_bright_spot_does_not_override_model(self):
        result = choose_reflection((20, 20), .05, (80, 20), .90)
        self.assertEqual(result.point, (80.0, 20.0))
        self.assertEqual(result.source, "reflection_model_disagrees")

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
        result = stabilizer.update((11, 10), HeadChoice(
            (-20, 10), .05, "learned_low_confidence_fallback", None))
        self.assertEqual(result.source, "temporal_learned_low_confidence_fallback")
        self.assertGreater(result.point[0], 10)

    def test_short_missing_gap_uses_body_relative_prediction(self):
        stabilizer = HeadTemporalStabilizer(max_gap_frames=2)
        stabilizer.update((10, 10), HeadChoice((15, 10), .8, "reflection", None))
        result = stabilizer.update((12, 10), HeadChoice(None, 0, "missing", None))
        self.assertEqual(result.source, "temporal_short_gap")
        self.assertEqual(result.point, (17., 10.))

    def test_long_missing_gap_clears_stale_direction(self):
        stabilizer = HeadTemporalStabilizer(max_gap_frames=2)
        stabilizer.update((10, 10), HeadChoice((15, 10), .8, "reflection", None))
        stabilizer.update(None, HeadChoice(None, 0, "missing", None))
        stabilizer.update(None, HeadChoice(None, 0, "missing", None))
        stabilizer.update(None, HeadChoice(None, 0, "missing", None))
        result = stabilizer.update((80, 80), HeadChoice(None, 0, "missing", None))
        self.assertIsNone(result.point)

    def test_annotation_selection_prioritizes_failures(self):
        rows = pd.DataFrame({"frame": [1, 2, 3], "head_x_cm": [1, 1, 8],
                             "head_y_cm": [1, 1, 8], "head_confidence": [.9, .2, .8],
                             "head_disagreement_px": [1, 30, 2],
                             "head_source": ["reflection", "learned_fallback", "reflection"]})
        self.assertEqual(select_frames(rows, set(), 1), [2])


if __name__ == "__main__":
    unittest.main()
