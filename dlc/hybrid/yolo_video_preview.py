"""In-memory YOLO prediction player for visual model validation."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSlider, QVBoxLayout,
)

from dlc.hybrid.device import resolve_ultralytics_device


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class VideoSurface(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(720, 480)
        self.setStyleSheet("background: #111827; border: 1px solid #334155;")
        self._image: QImage | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        self._image = QImage(
            rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888
        ).copy()
        self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        self.setPixmap(QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()


class YoloVideoPreviewDialog(QDialog):
    """Interactive player that predicts and draws the currently viewed frame only."""

    def __init__(
        self,
        video_path: Path,
        model_path: Path,
        settings: dict[str, Any],
        parent=None,
        *,
        model: Any = None,
        capture: Any = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.model_path = Path(model_path)
        self.settings = dict(settings)
        if not self.video_path.is_file() and capture is None:
            raise FileNotFoundError(self.video_path)
        if not self.model_path.is_file() and model is None:
            raise FileNotFoundError(self.model_path)
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(self.model_path))
        self.model = model
        self.capture = capture if capture is not None else cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total_frames = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        self.last_detection_count = 0
        self.last_inference_ms = 0.0
        self._raw_frame: np.ndarray | None = None
        self.device = resolve_ultralytics_device(self.settings.get("device", "auto"))

        self.setWindowTitle(f"YOLO video preview — {self.video_path.name}")
        self.resize(1280, 860)
        root = QVBoxLayout(self)
        self.surface = VideoSurface()
        root.addWidget(self.surface, 1)

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, self.total_frames - 1)
        self.timeline.sliderReleased.connect(lambda: self.seek(self.timeline.value()))
        root.addWidget(self.timeline)

        transport = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        previous = QPushButton("◀ Previous frame")
        previous.clicked.connect(lambda: self.step(-1))
        following = QPushButton("Next frame ▶")
        following.clicked.connect(lambda: self.step(1))
        transport.addWidget(previous)
        transport.addWidget(self.play_button)
        transport.addWidget(following)
        transport.addWidget(QLabel("Speed"))
        self.speed = QComboBox()
        for label, value in (("0.25×", .25), ("0.5×", .5), ("1×", 1.0), ("1.5×", 1.5), ("2×", 2.0)):
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._update_timer_interval)
        transport.addWidget(self.speed)
        self.fullscreen = QPushButton("Full screen")
        self.fullscreen.clicked.connect(self.toggle_fullscreen)
        transport.addWidget(self.fullscreen)
        transport.addStretch(1)
        root.addLayout(transport)

        options = QHBoxLayout()
        options.addWidget(QLabel("Confidence"))
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.001, 1.0)
        self.confidence.setDecimals(3)
        self.confidence.setSingleStep(.025)
        self.confidence.setValue(float(self.settings.get("confidence", .25)))
        options.addWidget(self.confidence)
        options.addWidget(QLabel("NMS IoU"))
        self.iou = QDoubleSpinBox()
        self.iou.setRange(.01, 1.0)
        self.iou.setDecimals(2)
        self.iou.setSingleStep(.05)
        self.iou.setValue(float(self.settings.get("iou", .70)))
        options.addWidget(self.iou)
        self.show_boxes = QCheckBox("Show boxes")
        self.show_boxes.setChecked(True)
        self.show_confidence = QCheckBox("Show class and confidence")
        self.show_confidence.setChecked(True)
        options.addWidget(self.show_boxes)
        options.addWidget(self.show_confidence)
        refresh = QPushButton("Re-run current frame")
        refresh.clicked.connect(self.rerender)
        options.addWidget(refresh)
        options.addStretch(1)
        root.addLayout(options)

        self.info = QLabel()
        self.info.setStyleSheet("color: #334155; padding: 4px;")
        root.addWidget(self.info)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(lambda: self.step(1, from_timer=True))
        self._update_timer_interval()
        self._shortcuts = [
            QShortcut(QKeySequence("Space"), self, activated=self.toggle_playback),
            QShortcut(QKeySequence("Left"), self, activated=lambda: self.step(-1)),
            QShortcut(QKeySequence("Right"), self, activated=lambda: self.step(1)),
        ]
        self.seek(0)

    def _update_timer_interval(self) -> None:
        speed = float(self.speed.currentData() or 1.0)
        if hasattr(self, "timer"):
            self.timer.setInterval(max(1, int(round(1000.0 / self.fps / speed))))

    def _predict(self, frame: np.ndarray) -> tuple[np.ndarray, int, float]:
        if not self.show_boxes.isChecked():
            return frame.copy(), 0, 0.0
        started = perf_counter()
        predictions = self.model.predict(
            source=frame,
            imgsz=int(self.settings.get("image_size", 640)),
            conf=float(self.confidence.value()),
            iou=float(self.iou.value()),
            device=self.device,
            verbose=False,
            save=False,
        )
        elapsed = (perf_counter() - started) * 1000.0
        result = predictions[0]
        boxes = getattr(result, "boxes", None)
        xyxy = _array(getattr(boxes, "xyxy", None)).reshape(-1, 4)
        scores = _array(getattr(boxes, "conf", None)).reshape(-1)
        classes = _array(getattr(boxes, "cls", None)).reshape(-1)
        names = getattr(result, "names", {0: "mouse"})
        drawn = frame.copy()
        thickness = max(2, int(round(min(frame.shape[:2]) / 300)))
        font_scale = max(.5, min(frame.shape[:2]) / 900)
        for index, coords in enumerate(xyxy):
            x1, y1, x2, y2 = np.rint(coords).astype(int)
            x1, x2 = sorted((int(np.clip(x1, 0, frame.shape[1] - 1)), int(np.clip(x2, 0, frame.shape[1] - 1))))
            y1, y2 = sorted((int(np.clip(y1, 0, frame.shape[0] - 1)), int(np.clip(y2, 0, frame.shape[0] - 1))))
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (50, 220, 80), thickness)
            if self.show_confidence.isChecked():
                class_id = int(classes[index]) if index < len(classes) else 0
                name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                score = float(scores[index]) if index < len(scores) else 0.0
                label = f"{name} {score:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                top = max(0, y1 - text_h - 8)
                cv2.rectangle(drawn, (x1, top), (min(frame.shape[1] - 1, x1 + text_w + 8), y1), (50, 220, 80), -1)
                cv2.putText(drawn, label, (x1 + 4, max(text_h, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 25, 15), thickness)
        return drawn, len(xyxy), elapsed

    def seek(self, frame_index: int) -> None:
        index = int(np.clip(frame_index, 0, self.total_frames - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.pause()
            return
        self.current_frame = index
        self._raw_frame = frame
        self.rerender()

    def rerender(self) -> None:
        if self._raw_frame is None:
            return
        try:
            shown, count, elapsed = self._predict(self._raw_frame)
        except Exception as exc:
            self.pause()
            QMessageBox.critical(self, "YOLO prediction error", str(exc))
            return
        self.last_detection_count, self.last_inference_ms = count, elapsed
        self.surface.set_frame(shown)
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_frame)
        self.timeline.blockSignals(False)
        seconds = self.current_frame / self.fps
        duration = self.total_frames / self.fps
        self.info.setText(
            f"Frame {self.current_frame + 1}/{self.total_frames} · "
            f"{seconds:.2f}/{duration:.2f} s · {self.width}×{self.height} @ {self.fps:.2f} fps · "
            f"detections={count} · inference={elapsed:.1f} ms · device={self.device}"
        )

    def step(self, delta: int, from_timer: bool = False) -> None:
        target = self.current_frame + delta
        if target < 0 or target >= self.total_frames:
            if from_timer:
                self.pause()
            return
        if not from_timer:
            self.pause()
        self.seek(target)

    def toggle_playback(self) -> None:
        if self.timer.isActive():
            self.pause()
        else:
            if self.current_frame >= self.total_frames - 1:
                self.seek(0)
            self.timer.start()
            self.play_button.setText("Pause")

    def pause(self) -> None:
        self.timer.stop()
        self.play_button.setText("Play")

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen.setText("Full screen")
        else:
            self.showFullScreen()
            self.fullscreen.setText("Exit full screen")

    def closeEvent(self, event) -> None:
        self.pause()
        self.capture.release()
        super().closeEvent(event)


def open_yolo_video_preview(video: Path, model: Path, settings: dict[str, Any], parent=None) -> None:
    YoloVideoPreviewDialog(video, model, settings, parent).exec()
