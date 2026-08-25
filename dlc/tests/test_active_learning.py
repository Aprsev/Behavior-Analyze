"""Tests for dataset reuse and low-confidence active-learning selection."""

from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

missing = [name for name in ("cv2", "numpy", "pandas") if find_spec(name) is None]
if missing:
    raise unittest.SkipTest("optional scientific dependencies missing: " + ", ".join(missing))

import cv2
import numpy as np
import pandas as pd

from dlc.hybrid.active_learning import Candidate, import_labeled_dataset, select_low_confidence


class ActiveLearningTests(unittest.TestCase):
    def test_native_dataset_import_is_content_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "old", root / "current"
            source.mkdir()
            image = source / "labeled.jpg"
            cv2.imwrite(str(image), np.zeros((60, 100, 3), np.uint8))
            pd.DataFrame([{
                "video": "old_video.avi", "frame": 12, "image": str(image),
                "x1": 10, "y1": 8, "x2": 70, "y2": 52,
                "source": "manual_review", "confidence": 1.0,
                "exclude": False, "reviewed": True,
            }]).to_csv(source / "box_labels.csv", index=False)
            cfg = {
                "yolo": {"dataset_dir": str(target)},
                "active_learning": {"import_dataset_dir": str(source)},
            }
            first = import_labeled_dataset(cfg)
            second = import_labeled_dataset(cfg)
            self.assertEqual(first["imported"], 1)
            self.assertEqual(second["imported"], 0)
            self.assertEqual(second["duplicates"], 1)
            imported = pd.read_csv(target / "box_labels.csv")
            self.assertEqual(len(imported), 1)
            self.assertTrue(Path(imported.iloc[0].image).is_file())

    def test_lowest_confidence_and_missing_detections_are_selected(self) -> None:
        candidates = [
            Candidate("a.avi", 0, 10.0, .9, (1, 1, 5, 5)),
            Candidate("a.avi", 10, 10.0, .2, (1, 1, 5, 5)),
            Candidate("b.avi", 20, 10.0, 0.0, None),
            Candidate("b.avi", 30, 10.0, .4, (1, 1, 5, 5)),
        ]
        selected = select_low_confidence(candidates, 2, .5)
        self.assertEqual([item.confidence for item in selected], [0.0, .2])

    def test_temporal_spacing_avoids_redundant_neighboring_frames(self) -> None:
        candidates = [
            Candidate("a.avi", 10, 10.0, .1, None),
            Candidate("a.avi", 11, 10.0, .2, None),
            Candidate("a.avi", 30, 10.0, .3, None),
        ]
        selected = select_low_confidence(candidates, 2, 1.0)
        self.assertEqual([item.frame for item in selected], [10, 30])


if __name__ == "__main__":
    unittest.main()
