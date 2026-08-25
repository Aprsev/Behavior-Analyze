#!/usr/bin/env python3
"""Primary Hybrid GUI with UTF-8, control-code-free live logging."""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("NO_COLOR", "1")
os.environ.setdefault("TERM", "dumb")

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QLineEdit

from dlc.hybrid.active_learning_gui import ActiveLearningWorkbench
from dlc.hybrid.log_text import sanitize_process_output


class CleanLogWorkbench(ActiveLearningWorkbench):
    def read_output(self) -> None:
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        text = sanitize_process_output(raw)
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()
        for line in text.splitlines():
            if not line.startswith("HYBRID_GUI_RESULT "):
                continue
            try:
                result = json.loads(line.removeprefix("HYBRID_GUI_RESULT "))
                if result.get("yolo_trained_model"):
                    field = self.fields["yolo.trained_model"]
                    assert isinstance(field, QLineEdit)
                    field.setText(result["yolo_trained_model"])
                    self.save_config()
            except Exception:
                pass


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = CleanLogWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
