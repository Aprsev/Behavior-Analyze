#!/usr/bin/env python3
import json
import unittest

from label_compat import as_bool, normalize_polygon, video_matches


class LabelCompatibilityTests(unittest.TestCase):
    def test_legacy_video_identifiers(self):
        target = "12026-08-12 160535.avi"
        self.assertTrue(video_matches(r"D:\old_host\data\12026-08-12 160535.avi", target))
        self.assertTrue(video_matches("160535", target))
        self.assertTrue(video_matches("12026-08-12_160535", target))
        self.assertFalse(video_matches("161211", target))

    def test_polygon_shapes_and_point_counts(self):
        nx2 = [[1, 2], [3, 4], [5, 6], [7, 8]]
        nx1x2 = [[[1, 2]], [[3, 4]], [[5, 6]]]
        flat = [1, 2, 3, 4, 5, 6]
        self.assertEqual(normalize_polygon(json.dumps(nx2)).shape, (4, 2))
        self.assertEqual(normalize_polygon(nx1x2).shape, (3, 2))
        self.assertEqual(normalize_polygon(flat).shape, (3, 2))

    def test_csv_boolean_strings(self):
        self.assertFalse(as_bool("False"))
        self.assertFalse(as_bool("0"))
        self.assertTrue(as_bool("True"))
        self.assertTrue(as_bool(1))


if __name__ == "__main__":
    unittest.main()
