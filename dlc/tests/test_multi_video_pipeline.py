"""Dependency-aware checks for video-grouped YOLO splitting."""

from importlib.util import find_spec
import unittest

missing = [name for name in ("cv2", "numpy", "pandas") if find_spec(name) is None]
if missing:
    raise unittest.SkipTest("optional scientific dependencies missing: " + ", ".join(missing))

from dlc.hybrid.multi_video_pipeline import _video_validation_set


class VideoSplitTests(unittest.TestCase):
    def test_validation_uses_complete_videos(self) -> None:
        videos = ["a.avi", "b.avi", "c.avi", "d.avi"]
        held_out = _video_validation_set(videos, 0.25, 42)
        self.assertEqual(len(held_out), 1)
        self.assertTrue(held_out.issubset(set(videos)))

    def test_single_video_uses_frame_level_fallback(self) -> None:
        self.assertEqual(_video_validation_set(["only.avi"], 0.2, 42), set())
