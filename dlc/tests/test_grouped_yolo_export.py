"""End-to-end test for leakage-safe YOLO train/validation export."""

from importlib.util import find_spec
from pathlib import Path
import json
import tempfile
import unittest

missing = [name for name in ("cv2", "numpy", "pandas") if find_spec(name) is None]
if missing:
    raise unittest.SkipTest("optional scientific dependencies missing: " + ", ".join(missing))

import cv2
import numpy as np
import pandas as pd

from dlc.hybrid.multi_video_pipeline import export_grouped_yolo_dataset


class GroupedYoloExportTests(unittest.TestCase):
    def test_frames_from_one_video_never_cross_train_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_images"
            source.mkdir()
            rows = []
            for video_index in range(4):
                video = f"video_{video_index}.avi"
                for frame_index in range(3):
                    image = source / f"v{video_index}_f{frame_index}.jpg"
                    cv2.imwrite(str(image), np.zeros((80, 120, 3), np.uint8))
                    rows.append({
                        "video": video,
                        "frame": frame_index,
                        "image": str(image),
                        "x1": 10,
                        "y1": 10,
                        "x2": 70,
                        "y2": 60,
                        "source": "traditional_background",
                        "confidence": 0.8,
                        "exclude": False,
                        "reviewed": True,
                    })
            pd.DataFrame(rows).to_csv(root / "box_labels.csv", index=False)
            cfg = {
                "yolo": {"dataset_dir": str(root), "validation_fraction": 0.25, "split_seed": 42},
                "video_sets": {"test": [{"video": "held_out.avi", "roi_json": "roi.json", "split": "test"}]},
            }
            export_grouped_yolo_dataset(cfg)
            manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_mode"], "whole_video")
            self.assertEqual(len(manifest["validation_videos"]), 1)
            self.assertEqual(manifest["external_test_videos"], ["held_out.avi"])
            val_prefix = "v" + manifest["validation_videos"][0].split("_")[1].split(".")[0] + "_"
            self.assertTrue(all(path.name.startswith(val_prefix) for path in (root / "images" / "val").glob("*.jpg")))
            self.assertFalse(any(path.name.startswith(val_prefix) for path in (root / "images" / "train").glob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
