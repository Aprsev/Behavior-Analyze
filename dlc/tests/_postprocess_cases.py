import unittest

import numpy as np
import pandas as pd

from dlc.postprocess import _fill_short_gaps, fuse_pose, normalize_dlc_dataframe, transform_points


class TestDlcPostprocess(unittest.TestCase):
    def test_reads_three_level_dlc_table_and_fuses_occluded_head(self):
        cols = pd.MultiIndex.from_product(
            [["DLC_model"], ["nose", "head_midpoint", "mouse_center", "mid_back"], ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(np.nan, index=range(2), columns=cols)
        values = {
            "nose": [[10, 10, 0.95], [11, 10, 0.10]],
            "head_midpoint": [[12, 10, 0.90], [13, 10, 0.90]],
            "mouse_center": [[20, 20, 0.98], [21, 20, 0.98]],
            "mid_back": [[21, 20, 0.90], [22, 20, 0.90]],
        }
        for bp, rows in values.items():
            df.loc[:, ("DLC_model", bp, "x")] = [r[0] for r in rows]
            df.loc[:, ("DLC_model", bp, "y")] = [r[1] for r in rows]
            df.loc[:, ("DLC_model", bp, "likelihood")] = [r[2] for r in rows]
        pose = normalize_dlc_dataframe(df)
        fused = fuse_pose(pose, 0.35)
        self.assertTrue(np.isfinite(fused["head"]).all())
        self.assertAlmostEqual(fused["head"][1, 0], 13.0)
        self.assertEqual(fused["body"][0, 0], 20.0)

    def test_selects_best_individual_in_four_level_table(self):
        cols = pd.MultiIndex.from_product(
            [["model"], ["mouse1", "mouse2"], ["mouse_center"], ["x", "y", "likelihood"]],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        df = pd.DataFrame([[1, 2, 0.2, 10, 20, 0.9]], columns=cols)
        pose = normalize_dlc_dataframe(df)
        self.assertEqual(pose.individual, "mouse2")
        self.assertEqual(pose.points["mouse_center"][0, 0], 10)

    def test_perspective_transform_keeps_nan(self):
        points = np.asarray([[1.0, 2.0], [np.nan, np.nan]])
        matrix = np.eye(3)
        actual = transform_points(points, matrix)
        np.testing.assert_allclose(actual[0], points[0])
        self.assertTrue(np.isnan(actual[1]).all())

    def test_only_complete_short_gaps_are_interpolated(self):
        values = np.asarray([[0.0, 0.0], [np.nan, np.nan], [2.0, 2.0],
                             [np.nan, np.nan], [np.nan, np.nan], [5.0, 5.0]])
        actual = _fill_short_gaps(values, max_gap_frames=1, median_window=1)
        np.testing.assert_allclose(actual[1], [1.0, 1.0])
        self.assertTrue(np.isnan(actual[3:5]).all())


if __name__ == "__main__":
    unittest.main()
