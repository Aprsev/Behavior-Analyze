#!/usr/bin/env python3
"""Primary GUI entry point including active YOLO label refinement."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from dlc.hybrid.active_learning_gui import ActiveLearningWorkbench


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = ActiveLearningWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
