#!/usr/bin/env python3
"""Primary Hybrid GUI with automatic unambiguous DLC project discovery."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QLineEdit

from dlc.hybrid import _project_result_gui as _base
from dlc.hybrid._project_result_gui import *  # noqa: F401,F403
from dlc.hybrid.project_config import resolve_project_config


class HybridWorkbench(_base.HybridWorkbench):
    def start_job(self, action: str, label: str) -> None:
        if action in _base.PROJECT_REQUIRED_ACTIONS:
            project = self._project_config_path()
            if project is None or not project.is_file():
                try:
                    project = resolve_project_config(self.collect())
                except FileNotFoundError:
                    pass
                else:
                    field = self.fields.get("project_config")
                    if isinstance(field, QLineEdit):
                        field.setText(str(project))
                        self.save_config()
        super().start_job(action, label)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = HybridWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
