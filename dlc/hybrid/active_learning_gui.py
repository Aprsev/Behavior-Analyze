"""Active-learning controls added to the path-safe Hybrid Workbench."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QMessageBox, QPushButton

from dlc.hybrid.path_safe_gui import PathSafeMultiVideoWorkbench


class ActiveLearningWorkbench(PathSafeMultiVideoWorkbench):
    def _boxes_tab(self):
        scroll = super()._boxes_tab()
        layout = scroll.widget().layout()
        box, form = self._group("Active learning from an existing YOLO model")
        self._line(form, "active_learning.import_dataset_dir", "Existing labeled dataset", browse="dir")
        self._multi_paths(
            form, "active_learning.new_videos", "New unlabeled training videos",
            "Videos (*.avi *.mp4 *.mov *.mkv *.m4v);;All files (*)", 90,
        )
        self._spin(form, "active_learning.candidate_frames_per_video", "Candidates scored per video", 300, 10, 100000)
        self._spin(form, "active_learning.frames_to_review", "Lowest-confidence frames to review", 80, 1, 100000)
        self._double(form, "active_learning.minimum_gap_sec", "Minimum spacing between selected frames (s)", .50, 0, 3600, 2)
        self._double(form, "active_learning.min_prediction_confidence", "Minimum prediction confidence", .001, 0, 1, 4)
        self._spin(form, "active_learning.fine_tune_epochs", "Fine-tuning epochs", 50, 1, 100000)
        self._double(form, "active_learning.learning_rate", "Fine-tuning initial learning rate", .001, .000001, 1, 6)
        self._line(form, "active_learning.run_name", "Fine-tuning run name", "mouse_detector_active")
        layout.insertWidget(layout.count() - 1, box)
        controls = QHBoxLayout()
        specs = [
            ("1 · Import old labels", "import_labeled_dataset"),
            ("2 · Mine low-confidence frames", "mine_active_frames"),
        ]
        for text, action in specs:
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, a=action, t=text: self.start_job(a, t))
            controls.addWidget(button)
        review = QPushButton("3 · Review queue frame-by-frame")
        review.clicked.connect(self.open_active_review)
        controls.addWidget(review)
        export = QPushButton("4 · Export combined dataset")
        export.clicked.connect(lambda: self.start_job("export_yolo", "Export combined YOLO dataset"))
        controls.addWidget(export)
        tune = QPushButton("5 · Fine-tune old best.pt")
        tune.setProperty("primary", True)
        tune.clicked.connect(lambda: self.start_job("fine_tune_yolo", "Fine-tune existing YOLO checkpoint"))
        controls.addWidget(tune)
        controls.addStretch(1)
        layout.insertLayout(layout.count() - 1, controls)
        layout.insertWidget(layout.count() - 1, self._callout(
            "Import is content-hash deduplicated. Mining uses the current Trained best.pt, prioritizes missing/low-confidence detections, and preserves the original model confidence after manual correction."
        ))
        return scroll

    def open_active_review(self) -> None:
        if not self.save_config():
            return
        try:
            from dlc.hybrid.active_review import open_active_review
            field = self.fields["yolo.dataset_dir"]
            assert isinstance(field, QLineEdit)
            dataset = Path(field.text()).expanduser()
            if not dataset.is_absolute():
                dataset = (self.config_path.parent / dataset).resolve()
            open_active_review(dataset / "box_labels.csv", self)
        except Exception as exc:
            QMessageBox.critical(self, "Active-learning review error", str(exc))
