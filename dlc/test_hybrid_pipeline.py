import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dlc.hybrid_pipeline import (
    _square_crop, export_yolo_dataset, load_box_labels,
    map_pose_table_to_source, update_box_label,
)
from dlc.postprocess import PoseTable


class TestHybridPipeline(unittest.TestCase):
    def test_crop_transform_maps_center_back_to_source(self):
        frame = np.zeros((300, 400, 3), np.uint8)
        crop, x0, y0, size = _square_crop(frame, np.asarray([100, 80, 200, 180]), 1.5)
        self.assertEqual(crop.shape[0], crop.shape[1])
        table = PoseTable(
            points={"mouse_center": np.asarray([[256.0, 256.0, 0.9]])},
            individual="", frame_count=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            transforms = Path(directory) / "transforms.csv"
            pd.DataFrame([{
                "x0": x0, "y0": y0, "crop_size": size,
                "output_size": 512, "detector_confidence": .9, "box_source": "yolo",
            }]).to_csv(transforms, index=False)
            mapped, _ = map_pose_table_to_source(table, transforms)
        self.assertAlmostEqual(mapped.points["mouse_center"][0, 0], x0 + size / 2, places=5)
        self.assertAlmostEqual(mapped.points["mouse_center"][0, 1], y0 + size / 2, places=5)

    def test_missing_crop_produces_missing_source_keypoint(self):
        table = PoseTable(
            points={"nose": np.asarray([[10.0, 20.0, 0.8]])}, individual="", frame_count=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            transforms = Path(directory) / "transforms.csv"
            pd.DataFrame([{
                "x0": np.nan, "y0": np.nan, "crop_size": np.nan,
                "output_size": 512, "detector_confidence": np.nan, "box_source": "missing",
            }]).to_csv(transforms, index=False)
            mapped, _ = map_pose_table_to_source(table, transforms)
        self.assertTrue(np.isnan(mapped.points["nose"][0, :2]).all())

    def test_manual_box_update_and_yolo_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_images"
            source.mkdir()
            rows = []
            for index in range(4):
                image = source / f"sample_{index}.jpg"
                cv2.imwrite(str(image), np.zeros((100, 200, 3), np.uint8))
                rows.append({
                    "video": "video.avi", "frame": index, "image": str(image),
                    "x1": 20, "y1": 10, "x2": 120, "y2": 70,
                    "source": "traditional_background", "confidence": .7,
                    "exclude": False, "reviewed": False,
                })
            labels = root / "box_labels.csv"
            pd.DataFrame(rows).to_csv(labels, index=False)
            update_box_label(labels, 0, (30, 20, 130, 80), False)
            changed = load_box_labels(labels).iloc[0]
            self.assertEqual(changed.source, "manual_review")
            self.assertTrue(changed.reviewed)
            data_yaml = export_yolo_dataset({"yolo": {
                "dataset_dir": str(root), "validation_fraction": .25, "split_seed": 4,
            }})
            self.assertTrue(data_yaml.is_file())
            label_files = list((root / "labels").rglob("*.txt"))
            self.assertEqual(len(label_files), 4)
            fields = label_files[0].read_text().split()
            self.assertEqual(fields[0], "0")
            self.assertTrue(all(0 <= float(value) <= 1 for value in fields[1:]))


if __name__ == "__main__":
    unittest.main()
