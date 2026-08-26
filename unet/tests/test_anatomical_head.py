#!/usr/bin/env python3
"""Synthetic tests for anatomical head/tail constraints."""
import unittest

try:
    import cv2
    import numpy as np
    from tracking.anatomical_head import AnatomicalHeadConstraint, clamp_choice_to_mask
    from tracking.head_fusion import HeadChoice
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class AnatomicalHeadTests(unittest.TestCase):
    def elongated_mask(self):
        mask = np.zeros((120, 180), np.uint8)
        cv2.ellipse(mask, (90, 60), (48, 18), 0, 0, 360, 255, -1)
        return mask

    def test_outside_candidate_is_clamped(self):
        mask = self.elongated_mask()
        choice = clamp_choice_to_mask(HeadChoice((165, 60), .8, "model", None), mask)
        x, y = np.rint(choice.point).astype(int)
        self.assertGreater(mask[y, x], 0)

    def test_reflection_wins_over_ambiguous_contained_branch(self):
        mask = self.elongated_mask()
        foreground = mask.copy()
        cv2.line(foreground, (42, 60), (12, 68), 255, 3)
        tracker = AnatomicalHeadConstraint()
        result = tracker.update((90, 60), mask, foreground,
                                HeadChoice(
                                    (42, 60), .9,
                                    "fused_reflection_primary_reflection_model_consensus",
                                    None))
        self.assertTrue(result.tail_detected)
        self.assertLess(result.choice.point[0], 90)

    def test_boundary_branch_is_not_tail(self):
        mask = self.elongated_mask()
        foreground = mask.copy()
        cv2.line(foreground, (138, 60), (179, 45), 255, 3)
        tracker = AnatomicalHeadConstraint()
        geometry = tracker.geometry(mask)
        tail = tracker.tail_endpoint(mask, foreground, geometry[2])
        self.assertIsNone(tail)

    def test_medium_reflection_needs_two_frames_to_flip_stale_endpoint(self):
        mask = self.elongated_mask()
        tracker = AnatomicalHeadConstraint()
        tracker.update((90, 60), mask, mask,
                       HeadChoice((132, 60), .8, "reflection", None))
        first = tracker.update((90, 60), mask, mask,
                               HeadChoice((48, 60), .5,
                                          "reflection_reflection_model", None))
        second = tracker.update((90, 60), mask, mask,
                                HeadChoice((48, 60), .5,
                                           "reflection_reflection_model", None))
        self.assertGreater(first.choice.point[0], 90)
        self.assertLess(second.choice.point[0], 90)

    def test_low_quality_disagreeing_reflection_cannot_single_frame_flip(self):
        mask = self.elongated_mask()
        tracker = AnatomicalHeadConstraint()
        tracker.update((90, 60), mask, mask,
                       HeadChoice((132, 60), .8,
                                  "reflection_reflection_model_consensus", None))
        result = tracker.update(
            (90, 60), mask, mask,
            HeadChoice((48, 60), .8,
                       "reflection_reflection_model_disagrees", 50.0))
        self.assertGreater(result.choice.point[0], 90)


if __name__ == "__main__":
    unittest.main()
