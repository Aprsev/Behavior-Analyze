"""Integration checks for standard YOLO import and checkpoint fine-tuning setup."""

from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

missing = [name for name in ("cv2", "numpy", "pandas") if find_spec(name) is None]
if missing:
    raise unittest.SkipTest("optional scientific dependencies missing: " + ", ".join(missing))

import cv2
import numpy as np
import pandas as pd

from dlc.hybrid.active_learning import import_labeled_dataset
from dlc.hybrid import hybrid_jobs


class ActiveLearningIntegrationTests(unittest.TestCase):
    def test_standard_yolo_directory_can_be_imported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "standard", root / "current"
            images, labels = source / "images" / "train", source / "labels" / "train"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            image = images / "mouse.jpg"
            cv2.imwrite(str(image), np.zeros((100, 200, 3), np.uint8))
            (labels / "mouse.txt").write_text("0 0.5 0.5 0.4 0.6\n", encoding="utf-8")
            report = import_labeled_dataset({
                "yolo": {"dataset_dir": str(target)},
                "active_learning": {"import_dataset_dir": str(source)},
            })
            self.assertEqual(report["mode"], "standard_yolo")
            imported = pd.read_csv(target / "box_labels.csv").iloc[0]
            self.assertAlmostEqual(float(imported.x1), 60.0)
            self.assertAlmostEqual(float(imported.y1), 20.0)
            self.assertAlmostEqual(float(imported.x2), 140.0)
            self.assertAlmostEqual(float(imported.y2), 80.0)

    def test_fine_tune_initializes_from_old_checkpoint_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "old_best.pt"
            checkpoint.write_bytes(b"checkpoint")
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "data.yaml").write_text("names:\n  0: mouse\n", encoding="utf-8")
            captured = {}

            class FakeYOLO:
                def __init__(self, model):
                    captured["model"] = model

                def train(self, **kwargs):
                    captured["kwargs"] = kwargs
                    return types.SimpleNamespace(save_dir=root / "active_run")

            fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
            cfg = {
                "yolo": {
                    "trained_model": str(checkpoint), "dataset_dir": str(dataset),
                    "image_size": 640, "batch_size": 4, "device": "cpu",
                    "workers": 0, "patience": 10, "split_seed": 42,
                },
                "active_learning": {
                    "fine_tune_epochs": 12, "learning_rate": .0005,
                    "run_name": "active_round_2",
                },
                "advanced": {"fine_tune_yolo": {}},
            }
            with patch.dict(sys.modules, {"ultralytics": fake_module}), patch.object(hybrid_jobs, "emit"):
                hybrid_jobs.job_fine_tune_yolo(cfg)
            self.assertEqual(captured["model"], str(checkpoint.resolve()))
            self.assertFalse(captured["kwargs"]["resume"])
            self.assertEqual(captured["kwargs"]["epochs"], 12)
            self.assertAlmostEqual(captured["kwargs"]["lr0"], .0005)
            self.assertEqual(captured["kwargs"]["data"], str((dataset / "data.yaml").resolve()))


if __name__ == "__main__":
    unittest.main()
