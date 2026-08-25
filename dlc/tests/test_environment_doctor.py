from __future__ import annotations

import unittest
from unittest.mock import patch

from dlc.tools import environment_doctor as doctor


class VersionTests(unittest.TestCase):
    def test_numeric_version_accepts_build_suffix(self) -> None:
        self.assertEqual(doctor.numeric_version("3.0.1+cu126"), (3, 0, 1))

    def test_version_range_has_exclusive_upper_bound(self) -> None:
        self.assertTrue(doctor.version_in_range("2.2.3", (2, 0), (3, 0)))
        self.assertFalse(doctor.version_in_range("3.0.0", (2, 0), (3, 0)))


class PlanTests(unittest.TestCase):
    def test_plan_uses_current_python_and_selected_torch_index(self) -> None:
        broken = [
            doctor.CheckResult(
                "DeepLabCut",
                "MISSING",
                None,
                "missing",
                "deeplabcut[gui,modelzoo]>=3,<4",
            )
        ]
        commands = doctor.build_install_commands(broken, "cu126", None, False, False)
        self.assertEqual(commands[0][:4], [doctor.sys.executable, "-m", "pip", "install"])
        self.assertEqual(commands[0][-1], doctor.TORCH_INDEXES["cu126"])
        self.assertIn("deeplabcut[gui,modelzoo]>=3,<4", commands[1])

    @patch.object(doctor, "installed_opencv_distributions")
    def test_opencv_cleanup_only_occurs_with_explicit_flag(self, installed) -> None:
        installed.return_value = {"opencv-python": "4.10.0"}
        broken = [
            doctor.CheckResult(
                "OpenCV contrib",
                "BROKEN",
                "4.10",
                "missing capability",
                "opencv-contrib-python>=4.8",
            )
        ]
        without_fix = doctor.build_install_commands(broken, "keep", None, False, False)
        with_fix = doctor.build_install_commands(broken, "keep", None, True, False)
        self.assertNotIn("uninstall", without_fix[0])
        self.assertIn("uninstall", with_fix[0])


if __name__ == "__main__":
    unittest.main()
