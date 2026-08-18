#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from label_compat import (as_bool, atomic_upsert_head, atomic_upsert_polygon,
                          normalize_polygon, video_matches)


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

    def test_atomic_old_format_update_preserves_other_rows(self):
        folder = Path(tempfile.mkdtemp())
        labels = folder / "labels.csv"
        pd.DataFrame([
            {"video": r"D:\old\video.avi", "frame": 1,
             "polygon_px": json.dumps([[[1, 2]], [[3, 4]], [[5, 6]]]), "exclude": False},
            {"video": "other.avi", "frame": 2,
             "polygon_px": json.dumps([[7, 8], [9, 10], [11, 12]]), "exclude": False},
        ]).to_csv(labels, index=False)
        backup = atomic_upsert_polygon(labels, "video.avi", 1,
                                       [[10, 20], [30, 40], [50, 60]], False)
        result = pd.read_csv(labels)
        self.assertEqual(len(result), 2)
        self.assertTrue(backup.is_file())
        row = result.loc[result.video == "video.avi"].iloc[0]
        self.assertEqual(normalize_polygon(row.polygon_px).tolist(),
                         [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

    def test_atomic_head_pair_update(self):
        folder = Path(tempfile.mkdtemp()); labels = folder / "heads.csv"
        atomic_upsert_head(labels, "video.avi", 12, .4, (4.2, 5.3), (4.0, 5.0))
        atomic_upsert_head(labels, "video.avi", 12, .4, (4.4, 5.5), (4.1, 5.1))
        result = pd.read_csv(labels)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.head_x_cm.iloc[0], 4.4)
        self.assertTrue(labels.with_suffix(".csv.bak").is_file())


if __name__ == "__main__":
    unittest.main()
