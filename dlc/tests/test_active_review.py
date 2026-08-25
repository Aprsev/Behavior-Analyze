"""Offscreen regression test for Save + Next in active-learning review."""

from importlib.util import find_spec
import os
from pathlib import Path
import tempfile
import unittest

missing = [name for name in ("cv2", "numpy", "pandas", "PySide6") if find_spec(name) is None]
if missing:
    raise unittest.SkipTest("optional GUI/scientific dependencies missing: " + ", ".join(missing))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from dlc.hybrid.active_review import ActiveLearningReviewDialog


class ActiveReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_save_next_persists_current_box_and_advances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(2):
                image = root / f"frame_{index}.jpg"
                cv2.imwrite(str(image), np.zeros((80, 120, 3), np.uint8))
                rows.append({
                    "video": "new.avi", "frame": index, "image": str(image),
                    "x1": 10, "y1": 10, "x2": 70, "y2": 60,
                    "source": "active_yolo", "confidence": .2,
                    "model_confidence": .2, "exclude": False, "reviewed": False,
                    "review_batch": "active_20260101_000000",
                })
            labels = root / "box_labels.csv"
            pd.DataFrame(rows).to_csv(labels, index=False)
            dialog = ActiveLearningReviewDialog(labels)
            dialog.save_next()
            self.assertEqual(dialog.position, 1)
            saved = pd.read_csv(labels)
            self.assertTrue(bool(saved.iloc[0].reviewed))
            self.assertEqual(saved.iloc[0].source, "manual_review")
            self.assertAlmostEqual(float(saved.iloc[0].model_confidence), .2)
            dialog.close()


if __name__ == "__main__":
    unittest.main()
