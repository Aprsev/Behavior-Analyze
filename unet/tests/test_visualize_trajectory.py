import unittest

import numpy as np
import pandas as pd

from tools.visualize_trajectory import add_orientation_columns


class OrientationTests(unittest.TestCase):
    def test_cardinal_angles_and_missing_frame(self):
        data = pd.DataFrame({
            "body_x_cm": [0, 0, 0, 0, 0], "body_y_cm": [0, 0, 0, 0, 0],
            "head_x_cm": [1, 0, -1, 0, np.nan],
            "head_y_cm": [0, 1, 0, -1, np.nan]})
        result = add_orientation_columns(data)
        np.testing.assert_allclose(result.head_angle_deg.iloc[:4], [0, 90, 180, -90])
        self.assertTrue(np.isnan(result.head_angle_deg.iloc[4]))

    def test_unwraps_boundary(self):
        radians = np.radians([170, 179, -179, -170])
        data = pd.DataFrame({"body_x_cm": 0, "body_y_cm": 0,
                             "head_x_cm": np.cos(radians),
                             "head_y_cm": np.sin(radians)})
        result = add_orientation_columns(data)
        np.testing.assert_allclose(result.head_angle_unwrapped_deg,
                                   [170, 179, 181, 190], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
