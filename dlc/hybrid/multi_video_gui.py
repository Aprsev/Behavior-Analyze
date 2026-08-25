"""Train/test dataset controls layered on the original English workbench."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from dlc.hybrid import _hybrid_gui_impl as base


class MultiVideoWorkbench(base.HybridWorkbench):
    def _multi_paths(self, form, key: str, label: str, file_filter: str, height: int = 90) -> None:
        container = QWidget()
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        editor = QPlainTextEdit()
        editor.setMinimumHeight(height)
        self.fields[key] = editor
        column.addWidget(editor)
        buttons = QHBoxLayout()
        add = QPushButton("Add files…")
        add.clicked.connect(lambda _=False, k=key, f=file_filter: self._add_paths(k, f))
        clear = QPushButton("Clear")
        clear.clicked.connect(editor.clear)
        buttons.addWidget(add)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        column.addLayout(buttons)
        form.addRow(label, container)

    def _add_paths(self, key: str, file_filter: str) -> None:
        values, _ = QFileDialog.getOpenFileNames(self, "Select files", str(self.config_path.parent), file_filter)
        if not values:
            return
        editor = self.fields[key]
        assert isinstance(editor, QPlainTextEdit)
        existing = [line.strip() for line in editor.toPlainText().splitlines() if line.strip()]
        editor.setPlainText("\n".join(dict.fromkeys([*existing, *values])))

    def _setup_tab(self):
        page, layout = self._page()
        layout.addWidget(self._callout(
            "Training and held-out testing are isolated by video. Test videos are never used for pseudo-box generation, YOLO training, or supervised DLC fine-tuning."
        ))
        train, form = self._group("Training dataset")
        self._multi_paths(form, "training_videos", "Training videos", "Videos (*.avi *.mp4 *.mov *.mkv *.m4v);;All files (*)", 115)
        self._multi_paths(form, "training_rois", "Training ROI JSONs", "JSON (*.json)", 80)
        layout.addWidget(train)
        layout.addWidget(self._callout(
            "Provide one shared ROI JSON, or one ROI per video in matching line order. Internal YOLO validation holds out complete training videos when two or more videos are available."
        ))
        test, tform = self._group("Held-out test dataset")
        self._multi_paths(tform, "test_videos", "Test videos", "Videos (*.avi *.mp4 *.mov *.mkv *.m4v);;All files (*)", 105)
        self._multi_paths(tform, "test_rois", "Test ROI JSONs", "JSON (*.json)", 75)
        layout.addWidget(test)
        common, cform = self._group("Arena, output, and DLC project")
        self._line(cform, "output_dir", "Analysis output directory", browse="dir")
        self._double(cform, "arena_width_cm", "Arena width (cm)", 25, .1, 1000, 2)
        self._double(cform, "arena_height_cm", "Arena height (cm)", 30, .1, 1000, 2)
        self._line(cform, "working_directory", "DLC project root", browse="dir")
        self._line(cform, "project_config", "DLC project config.yaml", browse="yaml")
        layout.addWidget(common)
        layout.addWidget(self._actions([("Check environment and all files", "hybrid_check")], "No files are changed by this check."))
        layout.addStretch(1)
        return self._scroll(page)

    def _hybrid_tab(self):
        scroll = super()._hybrid_tab()
        layout = scroll.widget().layout()
        layout.insertWidget(layout.count() - 1, self._callout(
            "Use the explicit buttons below for an auditable comparison. TRAIN outputs diagnose fit; HELD-OUT TEST outputs estimate generalization."
        ))
        layout.insertWidget(layout.count() - 1, self._actions([
            ("Run complete TRAIN-set pipeline", "full_hybrid_train"),
            ("Run complete HELD-OUT TEST pipeline", "full_hybrid_test"),
        ]))
        return scroll

    def load_config(self, path: Path) -> None:
        super().load_config(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        migrated = {
            "training_videos": data.get("training_videos") or data.get("videos") or ([data["video"]] if data.get("video") else []),
            "training_rois": data.get("training_rois") or ([data["roi_json"]] if data.get("roi_json") else []),
            "test_videos": data.get("test_videos") or [],
            "test_rois": data.get("test_rois") or [],
        }
        for key, values in migrated.items():
            widget = self.fields.get(key)
            if isinstance(widget, QPlainTextEdit):
                widget.setPlainText("\n".join(map(str, values)))

    def collect(self) -> dict:
        data = super().collect()
        training = data.get("training_videos", [])
        rois = data.get("training_rois", [])
        data["videos"] = training
        data["video"] = training[0] if training else ""
        data["roi_json"] = rois[0] if rois else ""
        return data


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = MultiVideoWorkbench()
    window.show()
    return app.exec()
