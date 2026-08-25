"""Offscreen regression tests for the in-memory YOLO video player."""

from importlib.util import find_spec
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

missing = [name for name in ("cv2", "numpy", "PySide6") if find_spec(name) is None]
if missing:
    raise unittest.SkipTest("optional GUI/scientific dependencies missing: " + ", ".join(missing))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from dlc.hybrid.yolo_video_preview import YoloVideoPreviewDialog


class FakeCapture:
    def __init__(self):
        self.frames = [np.zeros((80, 120, 3), np.uint8) for _ in range(3)]
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return {
            cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_FRAME_WIDTH: 120,
            cv2.CAP_PROP_FRAME_HEIGHT: 80,
        }.get(key, 0)

    def set(self, key, value):
        if key == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=np.asarray([[10, 10, 70, 60]], float),
            conf=np.asarray([.85], float),
            cls=np.asarray([0], float),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "mouse"})]


class YoloVideoPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_player_predicts_without_saving_and_supports_seeking(self) -> None:
        capture, model = FakeCapture(), FakeModel()
        dialog = YoloVideoPreviewDialog(
            Path("virtual.avi"), Path("virtual.pt"),
            {"device": "cpu", "image_size": 640, "confidence": .25, "iou": .7},
            model=model, capture=capture,
        )
        self.assertEqual(dialog.timeline.maximum(), 2)
        self.assertEqual(dialog.last_detection_count, 1)
        self.assertFalse(model.calls[0]["save"])
        self.assertFalse(model.calls[0]["verbose"])
        dialog.step(1)
        self.assertEqual(dialog.current_frame, 1)
        self.assertIn("Frame 2/3", dialog.info.text())
        self.assertFalse(dialog.surface.pixmap().isNull())
        dialog.close()
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
