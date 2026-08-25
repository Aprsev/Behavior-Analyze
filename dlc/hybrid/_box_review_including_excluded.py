#!/usr/bin/env python3
"""English contact-sheet viewer and draggable YOLO box editor."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from PySide6.QtCore import QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QCheckBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
        QPushButton, QScrollArea, QVBoxLayout, QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PySide6 is required for box review") from exc

from dlc.hybrid_pipeline import load_box_labels, update_box_label


def _pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy())


class BoxCanvas(QWidget):
    boxChanged = Signal(tuple)

    def __init__(self, image: np.ndarray, box: tuple[float, float, float, float] | None):
        super().__init__()
        self.image = image
        self.pixmap = _pixmap(image)
        self.box = box
        self.drag_mode: str | None = None
        self.drag_start = QPointF()
        self.original_box = box
        self.setMinimumSize(900, 600)
        self.setMouseTracking(True)

    def _display_rect(self) -> QRectF:
        width, height = self.image.shape[1], self.image.shape[0]
        scale = min(self.width() / width, self.height() / height)
        shown_w, shown_h = width * scale, height * scale
        return QRectF((self.width() - shown_w) / 2, (self.height() - shown_h) / 2, shown_w, shown_h)

    def _to_image(self, point: QPointF) -> QPointF:
        rect = self._display_rect()
        return QPointF(
            np.clip((point.x() - rect.x()) * self.image.shape[1] / rect.width(), 0, self.image.shape[1] - 1),
            np.clip((point.y() - rect.y()) * self.image.shape[0] / rect.height(), 0, self.image.shape[0] - 1),
        )

    def _to_widget_rect(self, box: tuple[float, float, float, float]) -> QRectF:
        rect = self._display_rect()
        sx, sy = rect.width() / self.image.shape[1], rect.height() / self.image.shape[0]
        x1, y1, x2, y2 = box
        return QRectF(rect.x() + x1 * sx, rect.y() + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111827"))
        painter.drawPixmap(self._display_rect(), self.pixmap, QRectF(self.pixmap.rect()))
        if self.box is None:
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Drag to create a mouse box")
            return
        rect = self._to_widget_rect(self.box)
        painter.setPen(QPen(QColor("#22c55e"), 3))
        painter.drawRect(rect)
        painter.setBrush(QColor("#f8fafc"))
        for corner in (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()):
            painter.drawEllipse(corner, 6, 6)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        point = self._to_image(event.position())
        self.drag_start = point
        self.original_box = self.box
        if self.box is None:
            self.box = (point.x(), point.y(), point.x(), point.y())
            self.drag_mode = "new"
            return
        rect = self._to_widget_rect(self.box)
        corners = {
            "tl": rect.topLeft(), "tr": rect.topRight(),
            "bl": rect.bottomLeft(), "br": rect.bottomRight(),
        }
        nearest = min(corners, key=lambda key: (corners[key] - event.position()).manhattanLength())
        if (corners[nearest] - event.position()).manhattanLength() <= 20:
            self.drag_mode = nearest
        elif rect.contains(event.position()):
            self.drag_mode = "move"
        else:
            self.box = (point.x(), point.y(), point.x(), point.y())
            self.drag_mode = "new"

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag_mode is None or self.box is None:
            return
        point = self._to_image(event.position())
        x1, y1, x2, y2 = self.box
        if self.drag_mode == "new":
            x1, y1, x2, y2 = self.drag_start.x(), self.drag_start.y(), point.x(), point.y()
        elif self.drag_mode == "move" and self.original_box is not None:
            ox1, oy1, ox2, oy2 = self.original_box
            dx, dy = point.x() - self.drag_start.x(), point.y() - self.drag_start.y()
            width, height = ox2 - ox1, oy2 - oy1
            x1 = np.clip(ox1 + dx, 0, self.image.shape[1] - width)
            y1 = np.clip(oy1 + dy, 0, self.image.shape[0] - height)
            x2, y2 = x1 + width, y1 + height
        else:
            if "l" in self.drag_mode: x1 = point.x()
            if "r" in self.drag_mode: x2 = point.x()
            if "t" in self.drag_mode: y1 = point.y()
            if "b" in self.drag_mode: y2 = point.y()
        self.box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            if x2 - x1 < 4 or y2 - y1 < 4:
                self.box = self.original_box
            else:
                self.boxChanged.emit(self.box)
        self.drag_mode = None
        self.update()


class BoxEditor(QDialog):
    def __init__(self, labels_csv: Path, row_index: int, parent=None):
        super().__init__(parent)
        self.labels_csv, self.row_index = labels_csv, row_index
        self.rows = load_box_labels(labels_csv)
        row = self.rows.iloc[row_index]
        image = cv2.imread(str(row.image))
        if image is None:
            raise RuntimeError(f"Cannot read {row.image}")
        values = row[["x1", "y1", "x2", "y2"]].to_numpy(float)
        box = tuple(values) if np.isfinite(values).all() else None
        self.initial_box = box
        self.setWindowTitle(f"Mouse box editor — frame {int(row.frame)}")
        self.resize(1200, 820)
        layout = QVBoxLayout(self)
        info = QLabel(
            f"{Path(row.video).name} · frame {int(row.frame)} · source={row.source} · confidence={float(row.confidence):.3f}\n"
            "Drag inside to move, drag a corner to resize, or drag outside to create a new box."
        )
        layout.addWidget(info)
        self.canvas = BoxCanvas(image, box)
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.exclude = QCheckBox("Exclude this sample")
        self.exclude.setChecked(bool(row.exclude))
        reset = QPushButton("Reset automatic box")
        reset.clicked.connect(self.reset_box)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save and close")
        save.clicked.connect(self.save)
        controls.addWidget(self.exclude)
        controls.addWidget(reset)
        controls.addStretch(1)
        controls.addWidget(cancel)
        controls.addWidget(save)
        layout.addLayout(controls)

    def reset_box(self) -> None:
        self.canvas.box = self.initial_box
        self.canvas.update()

    def save(self) -> None:
        if self.canvas.box is None and not self.exclude.isChecked():
            QMessageBox.warning(self, "Missing box", "Create a box or exclude this sample.")
            return
        update_box_label(self.labels_csv, self.row_index, self.canvas.box, self.exclude.isChecked())
        self.accept()


class BoxReviewDialog(QDialog):
    COLS, ROWS = 4, 4

    def __init__(self, labels_csv: Path, parent=None):
        super().__init__(parent)
        self.labels_csv = labels_csv
        self.page = 0
        self.setWindowTitle("YOLO mouse-box review")
        self.resize(1280, 850)
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("Automatic box labels — green=accepted, orange=unreviewed, red=excluded")
        title.setStyleSheet("font-size: 16px; font-weight: 600")
        self.stats = QLabel()
        prev_button = QPushButton("← Previous page")
        next_button = QPushButton("Next page →")
        prev_button.clicked.connect(lambda: self.move_page(-1))
        next_button.clicked.connect(lambda: self.move_page(1))
        header.addWidget(title)
        header.addWidget(self.stats, 1)
        header.addWidget(prev_button)
        header.addWidget(next_button)
        root.addLayout(header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        scroll.setWidget(self.content)
        root.addWidget(scroll, 1)
        close = QPushButton("Close review")
        close.clicked.connect(self.accept)
        root.addWidget(close)
        self.render()

    @property
    def page_size(self) -> int:
        return self.COLS * self.ROWS

    def move_page(self, delta: int) -> None:
        rows = load_box_labels(self.labels_csv)
        maximum = max(0, (len(rows) - 1) // self.page_size)
        self.page = int(np.clip(self.page + delta, 0, maximum))
        self.render()

    def render(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        rows = load_box_labels(self.labels_csv)
        reviewed, excluded = int(rows.reviewed.sum()), int(rows.exclude.sum())
        pages = max(1, math.ceil(len(rows) / self.page_size))
        self.stats.setText(f"{len(rows)} samples · {reviewed} reviewed · {excluded} excluded · page {self.page + 1}/{pages}")
        start = self.page * self.page_size
        for local, (index, row) in enumerate(rows.iloc[start:start + self.page_size].iterrows()):
            image = cv2.imread(str(row.image))
            if image is None:
                continue
            values = row[["x1", "y1", "x2", "y2"]].to_numpy(float)
            if np.isfinite(values).all():
                x1, y1, x2, y2 = np.rint(values).astype(int)
                color = (0, 0, 255) if row.exclude else ((0, 255, 0) if row.reviewed else (0, 170, 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, image.shape[1] // 400))
            cv2.putText(image, f"f{int(row.frame)} {row.source} {float(row.confidence):.2f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2)
            button = QPushButton()
            button.setMinimumSize(270, 175)
            button.setIcon(_pixmap(image).scaled(260, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            button.setIconSize(button.minimumSize())
            button.setToolTip("Click to edit this box")
            button.clicked.connect(lambda _=False, i=int(index): self.edit(i))
            self.grid.addWidget(button, local // self.COLS, local % self.COLS)

    def edit(self, row_index: int) -> None:
        try:
            editor = BoxEditor(self.labels_csv, row_index, self)
            editor.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Editor error", str(exc))
        self.render()


def open_box_review(labels_csv: Path, parent=None) -> None:
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Generate automatic boxes first: {labels_csv}")
    BoxReviewDialog(labels_csv, parent).exec()
