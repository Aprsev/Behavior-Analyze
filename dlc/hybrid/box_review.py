"""Public YOLO box review with sequential Save + Next editing."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QPushButton

from dlc.hybrid import _box_review_hidden_facade as _base
from dlc.hybrid._box_review_hidden_facade import *  # noqa: F401,F403
from dlc.hybrid._hybrid_pipeline_impl import load_box_labels


def next_visible_index(labels_csv: Path, current_index: int) -> int | None:
    """Return the next non-excluded CSV row after ``current_index``."""

    rows = _base._base._impl.visible_box_rows(load_box_labels(labels_csv))
    following = [int(index) for index in rows.index if int(index) > current_index]
    return following[0] if following else None


class BoxEditor(_base.BoxEditor):
    """Single-frame editor offering both close and sequential-save actions."""

    def __init__(self, labels_csv: Path, row_index: int, parent=None):
        super().__init__(labels_csv, row_index, parent)
        self.advance_after_save = False
        for button in self.findChildren(QPushButton):
            if button.text() == "Save and close":
                button.setText("Save + Close")
        save_next = QPushButton("Save + Next →")
        save_next.setDefault(True)
        save_next.clicked.connect(self.save_next)
        controls = self.layout().itemAt(self.layout().count() - 1).layout()
        controls.addWidget(save_next)

    def save_next(self) -> None:
        self.advance_after_save = True
        super().save()
        if self.result() != QDialog.Accepted:
            self.advance_after_save = False


class BoxReviewDialog(_base.BoxReviewDialog):
    def __init__(self, labels_csv: Path, parent=None):
        super().__init__(labels_csv, parent)
        for label in self.findChildren(QLabel):
            if label.text().startswith("Automatic box labels"):
                label.setText("Automatic box labels — green=accepted, orange=unreviewed")

    def edit(self, row_index: int) -> None:
        current = row_index
        while current is not None:
            try:
                editor = BoxEditor(self.labels_csv, current, self)
                accepted = editor.exec() == QDialog.Accepted
            except Exception as exc:
                QMessageBox.critical(self, "Editor error", str(exc))
                break
            self.render()
            if not accepted or not editor.advance_after_save:
                break
            current = next_visible_index(self.labels_csv, current)
            if current is None:
                QMessageBox.information(self, "Review complete", "No more available boxes remain.")


def open_box_review(labels_csv: Path, parent=None) -> None:
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Generate automatic boxes first: {labels_csv}")
    BoxReviewDialog(labels_csv, parent).exec()
