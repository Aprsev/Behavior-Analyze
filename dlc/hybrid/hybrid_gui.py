#!/usr/bin/env python3
"""Primary Hybrid GUI with chunk-safe plain terminal output."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from dlc.hybrid import _plain_terminal_gui_impl as _impl
from dlc.hybrid._plain_terminal_gui_impl import *  # noqa: F401,F403
from dlc.hybrid.terminal_stream import TerminalStreamSanitizer


class PlainTerminalWorkbench(_impl.PlainTerminalWorkbench):
    def __init__(self, parent=None):
        # The original HybridWorkbench constructor has no parent parameter.
        _impl.CleanLogWorkbench.__init__(self)
        self._terminal_stream = TerminalStreamSanitizer()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = PlainTerminalWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
