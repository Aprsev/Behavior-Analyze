"""Regression test for constructing the paginated box-review dialog."""

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

from dlc.hybrid.box_review import BoxReviewDialog


class BoxReviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_renders_first_page_without_injected_math(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.jpg"
            cv2.imwrite(str(image), np.zeros((80, 120, 3), np.uint8))
            labels = root / "box_labels.csv"
            pd.DataFrame([{
                "video": "mouse.avi", "frame": 0, "image": str(image),
                "x1": 10, "y1": 10, "x2": 80, "y2": 60,
                "source": "traditional_background", "confidence": 0.8,
                "exclude": False, "reviewed": False,
            }]).to_csv(labels, index=False)
            dialog = BoxReviewDialog(labels)
            self.assertIn("page 1/1", dialog.stats.text())
            dialog.close()


if __name__ == "__main__":
    unittest.main()
