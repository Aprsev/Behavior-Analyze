#!/usr/bin/env python3
"""Primary Hybrid GUI with an in-memory YOLO video preview player."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QWidget,
)

from dlc.hybrid import _terminal_gui_entry as _base
from dlc.hybrid._terminal_gui_entry import *  # noqa: F401,F403


class HybridWorkbench(_base.PlainTerminalWorkbench):
    def __init__(self):
        super().__init__()
        self._add_yolo_preview_controls()

    def _add_yolo_preview_controls(self) -> None:
        index = next(
            (i for i in range(self.tabs.count()) if "YOLO Training" in self.tabs.tabText(i)),
            -1,
        )
        if index < 0:
            return
        scroll = self.tabs.widget(index)
        page = scroll.widget()
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 3, 0, 3)
        preview = QPushButton("Preview trained YOLO on a video…")
        preview.setProperty("primary", True)
        preview.clicked.connect(self.open_yolo_video_preview)
        row.addWidget(preview)
        row.addStretch(1)
        page.layout().insertWidget(page.layout().count() - 1, row_widget)

    def open_yolo_video_preview(self) -> None:
        if self.process is not None:
            QMessageBox.warning(self, "Task already running", "Wait for training to finish before using GPU preview.")
            return
        if not self.save_config():
            return
        settings = self.collect().get("yolo", {})
        field = self.fields["yolo.trained_model"]
        assert isinstance(field, QLineEdit)
        model = Path(field.text()).expanduser()
        if not model.is_absolute():
            model = (self.config_path.parent / model).resolve()
        if not model.is_file():
            QMessageBox.warning(self, "Missing YOLO model", "Select a trained best.pt before opening video preview.")
            return
        video, _ = QFileDialog.getOpenFileName(
            self,
            "Select a video for YOLO visual validation",
            str(self.config_path.parent),
            "Videos (*.avi *.mp4 *.mov *.mkv *.m4v *.wmv);;All files (*)",
        )
        if not video:
            return
        try:
            from dlc.hybrid.yolo_video_preview import open_yolo_video_preview
            open_yolo_video_preview(Path(video), model, settings, self)
        except Exception as exc:
            QMessageBox.critical(self, "YOLO video preview error", str(exc))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = HybridWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
