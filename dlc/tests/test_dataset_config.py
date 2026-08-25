from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dlc.hybrid.dataset_config import load_hybrid_config, pair_video_rois


class DatasetConfigTests(unittest.TestCase):
    def test_one_roi_is_broadcast_to_multiple_videos(self) -> None:
        rows = pair_video_rois(["a.avi", "b.avi"], ["arena.json"], "train")
        self.assertEqual([row["roi_json"] for row in rows], ["arena.json", "arena.json"])

    def test_mismatched_roi_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one ROI per video"):
            pair_video_rois(["a.avi", "b.avi", "c.avi"], ["a.json", "b.json"], "test")

    def test_old_modular_config_keeps_pre_move_path_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "dlc" / "hybrid"
            config_dir.mkdir(parents=True)
            path = config_dir / "hybrid_config.json"
            path.write_text(json.dumps({
                "videos": ["../data/train.avi"],
                "roi_json": "../traditional/roi.json",
                "output_dir": "../results/hybrid",
                "yolo": {"dataset_dir": "../yolo/mouse"},
            }), encoding="utf-8")
            cfg = load_hybrid_config(path)
            self.assertEqual(Path(cfg["training_videos"][0]), root / "data" / "train.avi")
            self.assertEqual(Path(cfg["training_rois"][0]), root / "traditional" / "roi.json")


if __name__ == "__main__":
    unittest.main()
