"""Contact-sheet box editing supports sequential Save + Next."""

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
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from dlc.hybrid.box_review import BoxEditor, next_visible_index


class BoxReviewSaveNextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _labels(root: Path) -> Path:
        rows = []
        for index, excluded in enumerate((False, True, False)):
            image = root / f"frame_{index}.jpg"
            cv2.imwrite(str(image), np.zeros((80, 120, 3), np.uint8))
            rows.append({
                "video": "mouse.avi", "frame": index, "image": str(image),
                "x1": 10, "y1": 10, "x2": 70, "y2": 60,
                "source": "traditional_background", "confidence": .8,
                "exclude": excluded, "reviewed": excluded,
            })
        labels = root / "box_labels.csv"
        pd.DataFrame(rows).to_csv(labels, index=False)
        return labels

    def test_editor_exposes_both_save_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = self._labels(Path(directory))
            editor = BoxEditor(labels, 0)
            texts = {button.text() for button in editor.findChildren(QPushButton)}
            self.assertIn("Save + Close", texts)
            self.assertIn("Save + Next →", texts)
            editor.save_next()
            self.assertEqual(editor.result(), QDialog.Accepted)
            self.assertTrue(editor.advance_after_save)
            self.assertTrue(bool(pd.read_csv(labels).iloc[0].reviewed))

    def test_next_skips_excluded_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = self._labels(Path(directory))
            self.assertEqual(next_visible_index(labels, 0), 2)
            self.assertIsNone(next_visible_index(labels, 2))


if __name__ == "__main__":
    unittest.main()
