#!/usr/bin/env python3
"""Primary GUI entry point with isolated, path-safe train/test video sets."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from dlc.hybrid.path_safe_gui import PathSafeMultiVideoWorkbench


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = PathSafeMultiVideoWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
