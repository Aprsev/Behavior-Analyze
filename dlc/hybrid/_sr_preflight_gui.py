#!/usr/bin/env python3
"""Primary Hybrid GUI with SR model discovery and preflight checks."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QMessageBox, QSpinBox

from dlc.hybrid import _yolo_preview_gui as _base
from dlc.hybrid._yolo_preview_gui import *  # noqa: F401,F403
from dlc.hybrid.sr_model import NEURAL_METHODS, resolve_super_resolution_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SR_ACTIONS = {"prepare_hybrid", "full_hybrid", "full_hybrid_train", "full_hybrid_test"}


class HybridWorkbench(_base.HybridWorkbench):
    def __init__(self):
        super().__init__()
        method = self.fields.get("super_resolution.method")
        scale = self.fields.get("super_resolution.scale")
        if isinstance(method, QComboBox):
            method.currentTextChanged.connect(self._autofill_sr_model)
        if isinstance(scale, QSpinBox):
            scale.valueChanged.connect(self._autofill_sr_model)
        self._autofill_sr_model()

    def _autofill_sr_model(self, *_args) -> None:
        field = self.fields.get("super_resolution.model_path")
        if not isinstance(field, QLineEdit):
            return
        current = Path(field.text()).expanduser() if field.text().strip() else None
        if current is not None and current.is_file():
            return
        cfg = self.collect()
        try:
            found = resolve_super_resolution_model(
                cfg, repository_root=REPOSITORY_ROOT, config_path=self.config_path, require=False
            )
        except (ValueError, OSError):
            return
        if found is not None:
            field.setText(str(found))

    def start_job(self, action: str, label: str) -> None:
        if action in SR_ACTIONS:
            cfg = self.collect()
            method = str(cfg.get("super_resolution", {}).get("method", "bicubic")).lower()
            if method in NEURAL_METHODS:
                try:
                    model = resolve_super_resolution_model(
                        cfg, repository_root=REPOSITORY_ROOT,
                        config_path=self.config_path, require=True,
                    )
                except Exception as exc:
                    QMessageBox.warning(self, "Super-resolution model required", str(exc))
                    return
                field = self.fields.get("super_resolution.model_path")
                if isinstance(field, QLineEdit) and model is not None:
                    field.setText(str(model))
        super().start_job(action, label)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = HybridWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
