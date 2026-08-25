"""Path-normalizing GUI used during migration from the pre-modular layout."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLineEdit, QMessageBox, QPlainTextEdit

from dlc.hybrid.dataset_config import load_hybrid_config
from dlc.hybrid.multi_video_gui import MultiVideoWorkbench


class PathSafeMultiVideoWorkbench(MultiVideoWorkbench):
    def load_config(self, path: Path) -> None:
        super().load_config(path)
        try:
            normalized = load_hybrid_config(path)
        except Exception as exc:
            QMessageBox.critical(self, "Dataset configuration error", str(exc))
            return
        list_values = {
            "training_videos": normalized.get("training_videos", []),
            "training_rois": normalized.get("training_rois", []),
            "test_videos": normalized.get("test_videos", []),
            "test_rois": normalized.get("test_rois", []),
        }
        for key, values in list_values.items():
            widget = self.fields.get(key)
            if isinstance(widget, QPlainTextEdit):
                widget.setPlainText("\n".join(map(str, values)))
        scalar_values = {
            "output_dir": normalized.get("output_dir", ""),
            "working_directory": normalized.get("working_directory", ""),
            "project_config": normalized.get("project_config", ""),
            "yolo.dataset_dir": normalized.get("yolo", {}).get("dataset_dir", ""),
            "yolo.base_model": normalized.get("yolo", {}).get("base_model", ""),
            "yolo.trained_model": normalized.get("yolo", {}).get("trained_model", ""),
            "super_resolution.model_path": normalized.get("super_resolution", {}).get("model_path", ""),
        }
        for key, value in scalar_values.items():
            widget = self.fields.get(key)
            if isinstance(widget, QLineEdit) and value:
                widget.setText(str(value))
        if normalized.get("test_videos"):
            self.status.setText(
                f"Loaded {len(normalized['training_videos'])} training and "
                f"{len(normalized['test_videos'])} held-out test videos"
            )

    def collect(self) -> dict:
        data = super().collect()
        data["config_schema_version"] = 2
        return data
