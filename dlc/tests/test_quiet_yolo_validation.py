"""Standalone YOLO validation suppresses dynamic Ultralytics progress output."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dlc.hybrid import hybrid_jobs


class QuietYoloValidationTests(unittest.TestCase):
    def test_validation_forces_verbose_false_and_reports_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"weights")
            (root / "data.yaml").write_text("names:\n  0: mouse\n", encoding="utf-8")
            captured = {}

            class FakeYOLO:
                def __init__(self, model):
                    captured["model"] = model

                def val(self, **kwargs):
                    captured["kwargs"] = kwargs
                    box = SimpleNamespace(map=.388, map50=.887, mp=.794, mr=.866)
                    return SimpleNamespace(box=box)

            cfg = {
                "yolo": {
                    "trained_model": str(checkpoint), "dataset_dir": str(root),
                    "device": "cpu", "image_size": 640, "batch_size": 8,
                },
                "advanced": {"validate_yolo": {"verbose": True}},
            }
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
                hybrid_jobs.job_validate_yolo(cfg)
            self.assertEqual(captured["model"], str(checkpoint))
            self.assertFalse(captured["kwargs"]["verbose"])


if __name__ == "__main__":
    unittest.main()
