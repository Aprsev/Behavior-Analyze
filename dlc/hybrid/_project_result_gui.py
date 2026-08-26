#!/usr/bin/env python3
"""Primary Hybrid GUI with a guarded, persistent DLC project workflow."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox

from dlc.hybrid import _wheel_safe_gui as _base
from dlc.hybrid._wheel_safe_gui import *  # noqa: F401,F403
from dlc.hybrid.gui_results import parse_result_line

PROJECT_REQUIRED_ACTIONS = {
    "add_videos", "extract_frames", "label_frames", "check_labels",
    "build_dataset", "train", "evaluate", "analyze", "filter_predictions",
    "create_labeled_video", "plot_trajectories", "extract_outliers",
    "refine_labels", "merge_datasets",
}


class HybridWorkbench(_base.HybridWorkbench):
    """Persist a newly created DLC config and block out-of-order project actions."""

    def _project_config_path(self) -> Path | None:
        field = self.fields.get("project_config")
        if not isinstance(field, QLineEdit) or not field.text().strip():
            return None
        path = Path(field.text()).expanduser()
        return path if path.is_absolute() else (self.config_path.parent / path).resolve()

    def start_job(self, action: str, label: str) -> None:
        if action in PROJECT_REQUIRED_ACTIONS:
            project = self._project_config_path()
            if project is None or not project.is_file():
                QMessageBox.warning(
                    self,
                    "DLC project required",
                    "This action needs an existing DeepLabCut project config.yaml.\n\n"
                    "First click 'Create DLC project', or select an existing config.yaml "
                    "in Setup → DLC project config.yaml.",
                )
                index = next(
                    (i for i in range(self.tabs.count()) if "DLC Fine-tuning" in self.tabs.tabText(i)),
                    -1,
                )
                if index >= 0:
                    self.tabs.setCurrentIndex(index)
                return
        super().start_job(action, label)

    def _append_worker_text(self, text: str) -> None:
        super()._append_worker_text(text)
        for line in text.splitlines():
            try:
                result = parse_result_line(line)
            except (ValueError, TypeError):
                continue
            if not result or not result.get("project_config"):
                continue
            project = Path(str(result["project_config"])).expanduser().resolve()
            field = self.fields.get("project_config")
            if isinstance(field, QLineEdit):
                field.setText(str(project))
                self.save_config()
                self.status.setText(f"DLC project saved: {project.name}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = HybridWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
