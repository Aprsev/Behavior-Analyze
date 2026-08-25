"""Excluded audit rows never enter the generated YOLO dataset."""

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

from dlc.hybrid.multi_video_pipeline import export_grouped_yolo_dataset


class ExcludedExportTests(unittest.TestCase):
    def test_excluded_image_and_stale_label_are_absent_after_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_images"
            source.mkdir()
            rows = []
            for index, excluded in enumerate((True, False, False)):
                image = source / f"frame_{index}.jpg"
                cv2.imwrite(str(image), np.zeros((80, 120, 3), np.uint8))
                rows.append({
                    "video": "mouse.avi", "frame": index, "image": str(image),
                    "x1": 10, "y1": 10, "x2": 70, "y2": 60,
                    "source": "manual_review", "confidence": 1.0,
                    "exclude": excluded, "reviewed": True,
                })
            pd.DataFrame(rows).to_csv(root / "box_labels.csv", index=False)

            stale_images = root / "images" / "train"
            stale_labels = root / "labels" / "train"
            stale_images.mkdir(parents=True)
            stale_labels.mkdir(parents=True)
            (stale_images / "frame_0.jpg").write_bytes(b"stale")
            (stale_labels / "frame_0.txt").write_text("stale", encoding="utf-8")

            export_grouped_yolo_dataset({
                "yolo": {
                    "dataset_dir": str(root),
                    "validation_fraction": .5,
                    "split_seed": 42,
                },
                "video_sets": {"test": []},
            })

            exported = list((root / "images").rglob("frame_0.jpg"))
            labels = list((root / "labels").rglob("frame_0.txt"))
            self.assertEqual(exported, [])
            self.assertEqual(labels, [])


if __name__ == "__main__":
    unittest.main()
