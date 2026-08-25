#!/usr/bin/env python3
"""Primary Hybrid GUI with stateful plain-text subprocess logging."""

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

from dlc.hybrid._clean_log_gui import CleanLogWorkbench
from dlc.hybrid.terminal_stream import TerminalStreamSanitizer


class PlainTerminalWorkbench(CleanLogWorkbench):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._terminal_stream = TerminalStreamSanitizer()

    def _append_worker_text(self, text: str) -> None:
        if not text:
            return
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

    def read_output(self) -> None:
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_worker_text(self._terminal_stream.feed(raw))

    def job_finished(self, code: int, status) -> None:
        self.read_output()
        self._append_worker_text(self._terminal_stream.flush())
        self._terminal_stream = TerminalStreamSanitizer()
        super().job_finished(code, status)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = PlainTerminalWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
