"""Tests for matching existing DLC video settings by source stem."""

import unittest
from pathlib import Path

from dlc.hybrid.project_videos import _existing_settings


class ProjectVideoTests(unittest.TestCase):
    def test_reuses_crop_settings_for_same_video_stem(self) -> None:
        settings = _existing_settings(
            Path("D:/new/location/mouse01.avi"),
            [{"C:/old/location/mouse01.avi": {"crop": "1, 99, 2, 88"}}],
        )
        self.assertEqual(settings, {"crop": "1, 99, 2, 88"})

    def test_returns_none_for_new_video(self) -> None:
        self.assertIsNone(_existing_settings(Path("mouse02.avi"), [{"mouse01.avi": {}}]))


if __name__ == "__main__":
    unittest.main()
