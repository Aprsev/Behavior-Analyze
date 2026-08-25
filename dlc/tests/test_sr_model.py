"""Super-resolution weight discovery and compatibility checks."""

from pathlib import Path
import tempfile
import unittest

from dlc.hybrid.sr_model import expected_model_name, resolve_super_resolution_model


class SuperResolutionModelTests(unittest.TestCase):
    def test_expected_model_is_discovered_in_project_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "models" / "super_resolution"
            folder.mkdir(parents=True)
            model = folder / "EDSR_x4.pb"
            model.write_bytes(b"model")
            cfg = {"super_resolution": {"method": "edsr", "model_path": "", "scale": 4}}
            found = resolve_super_resolution_model(cfg, repository_root=root, require=True)
            self.assertEqual(found, model.resolve())
            self.assertEqual(cfg["super_resolution"]["model_path"], str(model.resolve()))

    def test_configured_relative_path_uses_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "configs" / "hybrid_config.json"
            model = config.parent / "weights" / "FSRCNN_x2.pb"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            cfg = {"super_resolution": {
                "method": "fsrcnn", "model_path": "weights/FSRCNN_x2.pb", "scale": 2,
            }}
            found = resolve_super_resolution_model(
                cfg, repository_root=root, config_path=config, require=True
            )
            self.assertEqual(found, model.resolve())

    def test_recognizable_mismatched_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "EDSR_x2.pb"
            model.write_bytes(b"model")
            cfg = {"super_resolution": {"method": "edsr", "model_path": str(model), "scale": 4}}
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_super_resolution_model(cfg, repository_root=Path(directory), require=True)

    def test_model_free_methods_need_no_weight(self) -> None:
        cfg = {"super_resolution": {"method": "bicubic", "model_path": "", "scale": 4}}
        self.assertIsNone(resolve_super_resolution_model(cfg, repository_root=Path.cwd(), require=True))
        self.assertEqual(expected_model_name("lapsrn", 4), "LapSRN_x4.pb")


if __name__ == "__main__":
    unittest.main()
