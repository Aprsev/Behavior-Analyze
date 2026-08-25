"""Safety checks preventing unreviewed or stale active labels from training."""

from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

missing = [name for name in ("numpy", "pandas") if find_spec(name) is None]
if missing:
    raise unittest.SkipTest("optional scientific dependencies missing: " + ", ".join(missing))

import pandas as pd

from dlc.hybrid.hybrid_jobs import pending_active_reviews, require_review_complete


class ActiveLearningGuardTests(unittest.TestCase):
    def test_unreviewed_active_rows_block_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            pd.DataFrame([{
                "video": "new.avi", "frame": 1, "image": "frame.jpg",
                "x1": 1, "y1": 1, "x2": 10, "y2": 10,
                "source": "active_yolo", "confidence": .2,
                "exclude": False, "reviewed": False,
                "review_batch": "active_20260101_000000",
            }]).to_csv(dataset / "box_labels.csv", index=False)
            cfg = {"yolo": {"dataset_dir": str(dataset)}}
            self.assertEqual(pending_active_reviews(cfg), 1)
            with self.assertRaisesRegex(RuntimeError, "still unsaved"):
                require_review_complete(cfg)

    def test_reviewed_active_rows_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            pd.DataFrame([{
                "video": "new.avi", "frame": 1, "image": "frame.jpg",
                "x1": 1, "y1": 1, "x2": 10, "y2": 10,
                "source": "manual_review", "confidence": 1.0,
                "exclude": False, "reviewed": True,
                "review_batch": "active_20260101_000000",
            }]).to_csv(dataset / "box_labels.csv", index=False)
            cfg = {"yolo": {"dataset_dir": str(dataset)}}
            require_review_complete(cfg)
            self.assertEqual(pending_active_reviews(cfg), 0)


if __name__ == "__main__":
    unittest.main()
