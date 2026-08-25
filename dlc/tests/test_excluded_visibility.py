"""Excluded labels stay in the audit CSV but leave review queues and exports."""

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
from dlc.hybrid.box_review import BoxReviewDialog


class ExcludedVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _dataset(root: Path) -> Path:
        rows = []
        for index, excluded in enumerate((True, False, False)):
            image = root / f"frame_{index}.jpg"
            cv2.imwrite(str(image), np.zeros((80, 120, 3), np.uint8))
            rows.append({
                "video": "mouse.avi", "frame": index, "image": str(image),
                "x1": 10, "y1": 10, "x2": 80, "y2": 60,
                "source": "active_yolo", "confidence": .2, "model_confidence": .2,
                "exclude": excluded, "reviewed": excluded,
                "review_batch": "active_20260101_000000",
            })
        labels = root / "box_labels.csv"
        pd.DataFrame(rows).to_csv(labels, index=False)
        return labels

    def test_contact_sheet_omits_excluded_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dialog = BoxReviewDialog(self._dataset(Path(directory)))
            self.assertEqual(dialog.grid.count(), 2)
            self.assertIn("2 available samples", dialog.stats.text())
            self.assertNotIn("excluded", dialog.stats.text().lower())
            dialog.close()

    def test_active_queue_omits_existing_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dialog = ActiveLearningReviewDialog(self._dataset(Path(directory)))
            self.assertEqual(dialog.indices, [1, 2])
            dialog.close()

    def test_save_next_removes_new_exclusion_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = self._dataset(Path(directory))
            dialog = ActiveLearningReviewDialog(labels)
            dialog.exclude.setChecked(True)
            dialog.save_next()
            self.assertEqual(dialog.indices, [2])
            self.assertEqual(dialog.position, 0)
            self.assertTrue(bool(pd.read_csv(labels).iloc[1].exclude))
            dialog.close()


if __name__ == "__main__":
    unittest.main()
