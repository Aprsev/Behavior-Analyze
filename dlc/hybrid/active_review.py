"""Sequential low-confidence frame editor with save-and-next workflow."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)

from dlc.hybrid.active_learning import latest_review_batch
from dlc.hybrid.box_review import BoxCanvas
from dlc.hybrid._hybrid_pipeline_impl import load_box_labels, update_box_label


class ActiveLearningReviewDialog(QDialog):
    def __init__(self, labels_csv: Path, parent=None, batch_id: str | None = None):
        super().__init__(parent)
        self.labels_csv = labels_csv
        self.batch_id = batch_id or latest_review_batch(labels_csv)
        rows = load_box_labels(labels_csv)
        self.indices = [
            int(index) for index, row in rows.iterrows()
            if str(row.get("review_batch", "")) == self.batch_id
        ]
        if not self.indices:
            raise ValueError(f"No frames in active-learning batch {self.batch_id}")
        self.position = 0
        self.canvas: BoxCanvas | None = None
        self.setWindowTitle(f"Active-learning box review — {self.batch_id}")
        self.resize(1250, 900)
        layout = QVBoxLayout(self)
        self.progress = QLabel()
        self.progress.setStyleSheet("font-size: 16px; font-weight: 600")
        self.details = QLabel()
        layout.addWidget(self.progress)
        layout.addWidget(self.details)
        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout, 1)
        controls = QHBoxLayout()
        previous = QPushButton("← Previous")
        previous.clicked.connect(self.previous)
        self.exclude = QCheckBox("Exclude this frame")
        save_close = QPushButton("Save + Close")
        save_close.clicked.connect(self.save_close)
        save_next = QPushButton("Save + Next →")
        save_next.setDefault(True)
        save_next.clicked.connect(self.save_next)
        controls.addWidget(previous)
        controls.addWidget(self.exclude)
        controls.addStretch(1)
        controls.addWidget(save_close)
        controls.addWidget(save_next)
        layout.addLayout(controls)
        self.load_current()

    def load_current(self) -> None:
        if self.canvas is not None:
            self.canvas.setParent(None)
            self.canvas.deleteLater()
        rows = load_box_labels(self.labels_csv)
        row = rows.iloc[self.indices[self.position]]
        image = cv2.imread(str(row.image))
        if image is None:
            raise RuntimeError(f"Cannot read {row.image}")
        values = row[["x1", "y1", "x2", "y2"]].to_numpy(float)
        box = tuple(values) if np.isfinite(values).all() else None
        self.canvas = BoxCanvas(image, box)
        self.canvas_layout.addWidget(self.canvas)
        self.exclude.setChecked(bool(row.exclude))
        confidence = float(row.get("model_confidence", row.confidence))
        reviewed = sum(bool(rows.iloc[index].reviewed) for index in self.indices)
        self.progress.setText(
            f"Frame {self.position + 1}/{len(self.indices)} · {reviewed}/{len(self.indices)} saved"
        )
        self.details.setText(
            f"{Path(str(row.video)).name} · frame {int(row.frame)} · model confidence={confidence:.4f}\n"
            "Drag inside the rectangle to move it, drag a corner to resize it, or drag outside to create one."
        )

    def _save(self) -> bool:
        assert self.canvas is not None
        if self.canvas.box is None and not self.exclude.isChecked():
            QMessageBox.warning(self, "Missing box", "Draw a box or exclude this frame.")
            return False
        update_box_label(
            self.labels_csv,
            self.indices[self.position],
            self.canvas.box,
            self.exclude.isChecked(),
        )
        return True

    def previous(self) -> None:
        if self.position > 0:
            self.position -= 1
            self.load_current()

    def save_close(self) -> None:
        if self._save():
            self.accept()

    def save_next(self) -> None:
        if not self._save():
            return
        if self.position + 1 >= len(self.indices):
            QMessageBox.information(self, "Review complete", "Every frame in this batch has been visited.")
            self.accept()
            return
        self.position += 1
        self.load_current()


def open_active_review(labels_csv: Path, parent=None) -> None:
    if not labels_csv.is_file():
        raise FileNotFoundError(f"No labels found: {labels_csv}")
    ActiveLearningReviewDialog(labels_csv, parent).exec()
