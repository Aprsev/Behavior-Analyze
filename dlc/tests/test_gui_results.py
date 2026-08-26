"""Structured worker-result protocol compatibility."""

import unittest

from dlc.hybrid.gui_results import parse_result_line


class GuiResultTests(unittest.TestCase):
    def test_parses_hybrid_result(self) -> None:
        self.assertEqual(
            parse_result_line('HYBRID_GUI_RESULT {"yolo_trained_model": "best.pt"}'),
            {"yolo_trained_model": "best.pt"},
        )

    def test_parses_legacy_dlc_project_result(self) -> None:
        self.assertEqual(
            parse_result_line('DLC_GUI_RESULT {"project_config": "D:/project/config.yaml"}'),
            {"project_config": "D:/project/config.yaml"},
        )

    def test_ignores_regular_log_line(self) -> None:
        self.assertIsNone(parse_result_line("Created project"))


if __name__ == "__main__":
    unittest.main()
