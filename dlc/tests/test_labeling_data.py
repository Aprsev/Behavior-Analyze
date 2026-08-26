"""Tests for extracted-frame verification and label-folder selection."""

import tempfile
import unittest
from pathlib import Path

from dlc.hybrid.labeling_data import choose_label_folder, extraction_summary


class LabelingDataTests(unittest.TestCase):
    def test_prefers_an_unlabeled_nonempty_folder(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            config = project / "config.yaml"
            config.write_text("Task: test", encoding="utf-8")
            done = project / "labeled-data" / "a_done"
            pending = project / "labeled-data" / "b_pending"
            done.mkdir(parents=True)
            pending.mkdir()
            (done / "img001.png").write_bytes(b"image")
            (done / "CollectedData_researcher.h5").write_bytes(b"labels")
            (pending / "img002.png").write_bytes(b"image")
            chosen, valid = choose_label_folder(config)
            self.assertEqual(chosen.path, pending.resolve())
            self.assertEqual(len(valid), 2)

    def test_rejects_empty_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            config = project / "config.yaml"
            config.write_text("Task: test", encoding="utf-8")
            (project / "labeled-data" / "empty").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "zero readable images"):
                extraction_summary(config)


if __name__ == "__main__":
    unittest.main()
