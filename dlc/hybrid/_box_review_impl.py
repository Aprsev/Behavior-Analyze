"""Contact-sheet review that hides samples already marked as excluded."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QPushButton

from dlc.hybrid import _box_review_including_excluded as _legacy
from dlc.hybrid._box_review_including_excluded import *  # noqa: F401,F403
from dlc.hybrid._hybrid_pipeline_impl import load_box_labels


def visible_box_rows(rows):
    """Return reviewable rows while preserving their CSV row indices."""

    return rows.loc[~rows.exclude].copy()


class BoxReviewDialog(_legacy.BoxReviewDialog):
    """Paginated contact sheet containing non-excluded samples only."""

    def __init__(self, labels_csv: Path, parent=None):
        super().__init__(labels_csv, parent)
        self.setWindowTitle("YOLO mouse-box review")

    def move_page(self, delta: int) -> None:
        rows = visible_box_rows(load_box_labels(self.labels_csv))
        maximum = max(0, (len(rows) - 1) // self.page_size)
        self.page = int(np.clip(self.page + delta, 0, maximum))
        self.render()

    def render(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        rows = visible_box_rows(load_box_labels(self.labels_csv))
        pages = max(1, math.ceil(len(rows) / self.page_size))
        self.page = min(self.page, pages - 1)
        reviewed = int(rows.reviewed.sum()) if len(rows) else 0
        self.stats.setText(
            f"{len(rows)} available samples · {reviewed} reviewed · page {self.page + 1}/{pages}"
        )
        start = self.page * self.page_size
        for local, (index, row) in enumerate(rows.iloc[start:start + self.page_size].iterrows()):
            image = cv2.imread(str(row.image))
            if image is None:
                continue
            values = row[["x1", "y1", "x2", "y2"]].to_numpy(float)
            if np.isfinite(values).all():
                x1, y1, x2, y2 = np.rint(values).astype(int)
                color = (0, 255, 0) if row.reviewed else (0, 170, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, image.shape[1] // 400))
            cv2.putText(
                image, f"f{int(row.frame)} {row.source} {float(row.confidence):.2f}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2,
            )
            button = QPushButton()
            button.setMinimumSize(270, 175)
            button.setIcon(
                _legacy._pixmap(image).scaled(260, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            button.setIconSize(button.minimumSize())
            button.setToolTip("Click to edit this box")
            button.clicked.connect(lambda _=False, i=int(index): self.edit(i))
            self.grid.addWidget(button, local // self.COLS, local % self.COLS)

    def edit(self, row_index: int) -> None:
        try:
            editor = _legacy.BoxEditor(self.labels_csv, row_index, self)
            editor.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Editor error", str(exc))
        # Saving an exclusion immediately removes that sample from this sheet.
        self.render()


def open_box_review(labels_csv: Path, parent=None) -> None:
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Generate automatic boxes first: {labels_csv}")
    BoxReviewDialog(labels_csv, parent).exec()
