#!/usr/bin/env python3
"""Primary Hybrid GUI with wheel-safe parameter controls."""

from __future__ import annotations

import sys

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QScrollArea,
)

from dlc.hybrid import _sr_preflight_gui as _base
from dlc.hybrid._sr_preflight_gui import *  # noqa: F401,F403


class HybridWorkbench(_base.HybridWorkbench):
    """Prevent accidental wheel edits while keeping page scrolling usable."""

    def __init__(self):
        super().__init__()
        controls = self.findChildren(QAbstractSpinBox) + self.findChildren(QComboBox)
        for control in controls:
            control.installEventFilter(self)

    def eventFilter(self, watched, event):
        if isinstance(watched, (QAbstractSpinBox, QComboBox)) and event.type() == QEvent.Wheel:
            parent = watched.parentWidget()
            while parent is not None and not isinstance(parent, QScrollArea):
                parent = parent.parentWidget()
            if isinstance(parent, QScrollArea):
                delta = event.angleDelta()
                if abs(delta.y()) >= abs(delta.x()):
                    bar = parent.verticalScrollBar()
                    bar.setValue(bar.value() - delta.y())
                else:
                    bar = parent.horizontalScrollBar()
                    bar.setValue(bar.value() - delta.x())
            event.accept()
            return True
        return super().eventFilter(watched, event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = HybridWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
