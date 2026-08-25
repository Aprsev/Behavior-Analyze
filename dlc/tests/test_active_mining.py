"""Synthetic-video integration test for low-confidence frame mining."""

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

from dlc.hybrid.active_learning import mine_low_confidence_frames


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class ActiveMiningTests(unittest.TestCase):
    def test_mining_writes_requested_low_confidence_review_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "new.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
            self.assertTrue(writer.isOpened())
            for index in range(20):
                writer.write(np.full((64, 96, 3), index * 5, np.uint8))
            writer.release()
            checkpoint = root / "old_best.pt"
            checkpoint.write_bytes(b"checkpoint")
            dataset = root / "dataset"
            calls = {"count": 0}

            class FakeYOLO:
                def __init__(self, _model):
                    pass

                def predict(self, _frame, **_kwargs):
                    calls["count"] += 1
                    confidence = calls["count"] / 100
                    boxes = types.SimpleNamespace(
                        xyxy=FakeTensor([[10, 8, 70, 55]]),
                        conf=FakeTensor([confidence]),
                        cls=FakeTensor([0]),
                    )
                    boxes.__len__ = lambda self: 1
                    return [types.SimpleNamespace(boxes=BoxCollection(boxes))]

            class BoxCollection:
                def __init__(self, values):
                    self.xyxy, self.conf, self.cls = values.xyxy, values.conf, values.cls

                def __len__(self):
                    return 1

            cfg = {
                "yolo": {
                    "trained_model": str(checkpoint), "dataset_dir": str(dataset),
                    "iou": .7, "image_size": 640, "device": "cpu",
                },
                "active_learning": {
                    "new_videos": [str(video)], "candidate_frames_per_video": 10,
                    "frames_to_review": 3, "minimum_gap_sec": 0,
                    "min_prediction_confidence": .001,
                },
            }
            with patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=FakeYOLO)}):
                report = mine_low_confidence_frames(cfg)
            self.assertEqual(report["selected"], 3)
            self.assertEqual(report["candidates_scored"], 10)
            labels = pd.read_csv(dataset / "box_labels.csv")
            self.assertEqual(len(labels), 3)
            self.assertTrue((~labels.reviewed).all())
            self.assertEqual(sorted(labels.model_confidence.round(2).tolist()), [.01, .02, .03])
            self.assertTrue(all(Path(value).is_file() for value in labels.image))


if __name__ == "__main__":
    unittest.main()
