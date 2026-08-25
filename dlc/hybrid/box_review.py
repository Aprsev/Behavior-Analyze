"""Public contact-sheet review with excluded samples hidden."""

from pathlib import Path

from PySide6.QtWidgets import QLabel

from dlc.hybrid import _box_review_public_base as _base
from dlc.hybrid._box_review_public_base import *  # noqa: F401,F403


class BoxReviewDialog(_base.BoxReviewDialog):
    def __init__(self, labels_csv: Path, parent=None):
        super().__init__(labels_csv, parent)
        for label in self.findChildren(QLabel):
            if label.text().startswith("Automatic box labels"):
                label.setText("Automatic box labels — green=accepted, orange=unreviewed")


def open_box_review(labels_csv: Path, parent=None) -> None:
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Generate automatic boxes first: {labels_csv}")
    BoxReviewDialog(labels_csv, parent).exec()
