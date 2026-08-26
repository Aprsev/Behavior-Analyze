"""DLC project config discovery safeguards."""

import tempfile
import unittest
from pathlib import Path

from dlc.hybrid.project_config import resolve_project_config


class ProjectConfigTests(unittest.TestCase):
    def test_discovers_one_matching_project(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "mouse_occlusion-researcher-2026-08-26" / "config.yaml"
            config.parent.mkdir()
            config.write_text("Task: mouse_occlusion", encoding="utf-8")
            cfg = {
                "working_directory": str(root),
                "project": {"task": "mouse_occlusion", "experimenter": "researcher"},
            }
            self.assertEqual(resolve_project_config(cfg), config.resolve())
            self.assertEqual(cfg["project_config"], str(config.resolve()))

    def test_rejects_ambiguous_matching_projects(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for suffix in ("a", "b"):
                config = root / f"mouse_occlusion-researcher-{suffix}" / "config.yaml"
                config.parent.mkdir()
                config.write_text("Task: mouse_occlusion", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Multiple matching"):
                resolve_project_config({"working_directory": str(root), "project": {}})


if __name__ == "__main__":
    unittest.main()
