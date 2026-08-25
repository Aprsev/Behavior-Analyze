#!/usr/bin/env python3
"""All-English GUI for the YOLO -> super-resolution -> DLC workflow."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
    from PySide6.QtGui import QDesktopServices, QFont, QTextCursor
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
        QScrollArea, QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PySide6 is missing. Install dlc/requirements-hybrid.txt.") from exc

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "hybrid_config.json"
EXAMPLE_CONFIG = HERE / "config.hybrid.example.json"


def nested_get(data: dict, key: str, default=None):
    value: Any = data
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def nested_set(data: dict, key: str, value: Any) -> None:
    target = data
    parts = key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


class HybridWorkbench(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mouse Pose Hybrid Workbench — YOLO · Super-Resolution · DeepLabCut")
        self.resize(1380, 900)
        self.fields: dict[str, QWidget] = {}
        self.process: QProcess | None = None
        self.config_path = DEFAULT_CONFIG
        self._build()
        self._style()
        self.load_config(DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else EXAMPLE_CONFIG)

    def _build(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 14, 18, 14)
        header = QHBoxLayout()
        headings = QVBoxLayout()
        title = QLabel("Mouse Pose Hybrid Workbench")
        title.setObjectName("title")
        subtitle = QLabel("Background pseudo-labels → reviewed YOLO boxes → super-resolution crops → DeepLabCut pose")
        subtitle.setObjectName("subtitle")
        headings.addWidget(title); headings.addWidget(subtitle)
        header.addLayout(headings, 1)
        self.config_label = QLabel()
        self.config_label.setObjectName("configPath")
        header.addWidget(self.config_label)
        for text, callback in (("Load", self.choose_config), ("Save", self.save_config), ("Save as", self.save_as)):
            button = QPushButton(text); button.clicked.connect(callback); header.addWidget(button)
        outer.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._setup_tab(), "1 · Setup")
        self.tabs.addTab(self._boxes_tab(), "2 · YOLO Labels")
        self.tabs.addTab(self._yolo_tab(), "3 · YOLO Training")
        self.tabs.addTab(self._hybrid_tab(), "4 · SR + DLC Inference")
        self.tabs.addTab(self._dlc_training_tab(), "5 · DLC Fine-tuning")
        self.tabs.addTab(self._tuning_tab(), "6 · Failure Mining")
        self.tabs.addTab(self._advanced_tab(), "Advanced API")
        self.tabs.addTab(self._log_tab(), "Live Log")
        outer.addWidget(self.tabs, 1)

        status_row = QHBoxLayout()
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.progress = QProgressBar(); self.progress.setRange(0, 1); self.progress.setValue(0); self.progress.setTextVisible(False)
        self.stop_button = QPushButton("Stop current task"); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self.stop_job)
        status_row.addWidget(self.status, 1); status_row.addWidget(self.progress, 2); status_row.addWidget(self.stop_button)
        outer.addLayout(status_row)
        self.setCentralWidget(central)

    def _page(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(8, 12, 8, 12); layout.setSpacing(12)
        return page, layout

    def _scroll(self, page: QWidget) -> QScrollArea:
        scroll = QScrollArea(); scroll.setFrameShape(QFrame.NoFrame); scroll.setWidgetResizable(True); scroll.setWidget(page)
        return scroll

    def _group(self, title: str):
        box = QGroupBox(title); form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return box, form

    def _line(self, form, key, label, default="", browse=None):
        widget = QLineEdit(default); self.fields[key] = widget
        if browse:
            row = QHBoxLayout(); row.addWidget(widget, 1)
            button = QPushButton("Browse…"); button.clicked.connect(lambda _=False, w=widget, m=browse: self.browse(w, m)); row.addWidget(button)
            form.addRow(label, row)
        else: form.addRow(label, widget)
        return widget

    def _text(self, form, key, label, default="", height=90):
        widget = QPlainTextEdit(default); widget.setMinimumHeight(height); self.fields[key] = widget; form.addRow(label, widget); return widget

    def _spin(self, form, key, label, default, low=0, high=1_000_000):
        widget = QSpinBox(); widget.setRange(low, high); widget.setValue(default); self.fields[key] = widget; form.addRow(label, widget); return widget

    def _double(self, form, key, label, default, low=0.0, high=1.0, decimals=3):
        widget = QDoubleSpinBox(); widget.setRange(low, high); widget.setDecimals(decimals); widget.setSingleStep(0.05); widget.setValue(default)
        self.fields[key] = widget; form.addRow(label, widget); return widget

    def _check(self, form, key, label, default):
        widget = QCheckBox(); widget.setChecked(default); self.fields[key] = widget; form.addRow(label, widget); return widget

    def _combo(self, form, key, label, values, default):
        widget = QComboBox(); widget.addItems(values); widget.setCurrentText(default); self.fields[key] = widget; form.addRow(label, widget); return widget

    def _actions(self, specs, hint=""):
        widget = QWidget(); row = QHBoxLayout(widget); row.setContentsMargins(0, 3, 0, 3)
        primary = {"generate_boxes", "train_yolo", "full_hybrid", "build_dataset", "train", "analyze"}
        for text, action in specs:
            button = QPushButton(text); button.setProperty("primary", action in primary)
            button.clicked.connect(lambda _=False, a=action, t=text: self.start_job(a, t)); row.addWidget(button)
        row.addStretch(1)
        if hint:
            label = QLabel(hint); label.setWordWrap(True); label.setObjectName("hint"); row.addWidget(label, 2)
        return widget

    def _callout(self, text, warning=False):
        label = QLabel(text); label.setWordWrap(True); label.setObjectName("warning" if warning else "callout"); return label

    def _setup_tab(self):
        page, layout = self._page()
        layout.addWidget(self._callout(
            "All paths and parameters are saved in one JSON file. Long-running jobs use child processes, so CUDA training and interactive labeling do not freeze this window."
        ))
        box, form = self._group("Videos, arena, and output")
        self._text(form, "videos", "Source videos (one per line)", height=105)
        self._line(form, "roi_json", "Four-corner ROI JSON", browse="json")
        self._line(form, "output_dir", "Analysis output directory", browse="dir")
        self._double(form, "arena_width_cm", "Arena width (cm)", 25, .1, 1000, 2)
        self._double(form, "arena_height_cm", "Arena height (cm)", 30, .1, 1000, 2)
        self._line(form, "working_directory", "DLC project root", browse="dir")
        self._line(form, "project_config", "DLC project config.yaml", browse="yaml")
        layout.addWidget(box)
        layout.addWidget(self._actions([("Check environment and files", "hybrid_check")], "No files are changed by this check."))
        layout.addWidget(self._callout(
            "Recommended order: generate boxes → review every uncertain box → export dataset → train/validate YOLO → prepare SR crops → run DLC → inverse-map and inspect final output."
        ))
        layout.addStretch(1); return self._scroll(page)

    def _boxes_tab(self):
        page, layout = self._page()
        box, form = self._group("Traditional background-subtraction pseudo-labels")
        self._line(form, "yolo.dataset_dir", "YOLO dataset directory", browse="dir")
        self._spin(form, "yolo.samples_per_video", "Candidate frames per video", 80, 2, 100000)
        self._spin(form, "yolo.background_samples", "Background calibration frames", 61, 3, 1001)
        self._double(form, "yolo.background_percentile", "Bright-floor percentile", 85, 50, 100, 1)
        self._double(form, "yolo.background_threshold", "Difference threshold (0=automatic)", 0, 0, 255, 2)
        self._double(form, "yolo.auto_box_padding", "Automatic box padding ratio", .15, 0, 2, 2)
        self._double(form, "yolo.validation_fraction", "Validation fraction", .20, .05, .8, 2)
        self._spin(form, "yolo.split_seed", "Deterministic split seed", 42, 0, 2_147_483_647)
        layout.addWidget(box)
        controls = QHBoxLayout()
        generate = QPushButton("1 · Generate automatic boxes"); generate.setProperty("primary", True); generate.clicked.connect(lambda: self.start_job("generate_boxes", "Generate automatic boxes"))
        review = QPushButton("2 · Review / edit all boxes"); review.clicked.connect(self.open_box_review)
        export = QPushButton("3 · Export reviewed YOLO dataset"); export.clicked.connect(lambda: self.start_job("export_yolo", "Export YOLO dataset"))
        controls.addWidget(generate); controls.addWidget(review); controls.addWidget(export); controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self._callout(
            "The contact sheet displays every label. Click a thumbnail to drag, resize, replace, or exclude its rectangle. Labels keep their source, automatic confidence, and review state for auditing."
        ))
        layout.addStretch(1); return self._scroll(page)

    def _yolo_tab(self):
        page, layout = self._page()
        box, form = self._group("Ultralytics mouse detector")
        self._line(form, "yolo.base_model", "Pretrained base model", "yolo26n.pt", browse="model_optional")
        self._line(form, "yolo.trained_model", "Trained best.pt", browse="model")
        self._line(form, "yolo.run_name", "Run name", "mouse_detector")
        self._combo(form, "yolo.device", "Device", ["auto", "0", "1", "cpu", "mps"], "auto")
        self._spin(form, "yolo.epochs", "Epochs", 100, 1, 100000)
        self._spin(form, "yolo.image_size", "Training / inference image size", 640, 64, 4096)
        self._spin(form, "yolo.batch_size", "Batch size", 8, 1, 4096)
        self._spin(form, "yolo.workers", "Data-loader workers", 4, 0, 128)
        self._spin(form, "yolo.patience", "Early-stop patience", 30, 0, 10000)
        self._double(form, "yolo.confidence", "Inference confidence", .25, 0, 1)
        self._double(form, "yolo.iou", "NMS IoU", .70, 0, 1)
        self._double(form, "yolo.crop_scale", "DLC crop scale around box", 1.50, 1, 5, 2)
        self._double(form, "yolo.max_fallback_sec", "Maximum temporal box fallback (s)", .30, 0, 10, 2)
        layout.addWidget(box)
        layout.addWidget(self._actions([
            ("Train YOLO", "train_yolo"), ("Validate YOLO", "validate_yolo")
        ], "Training writes plots, metrics, last.pt, and best.pt under dataset_dir/runs/run_name."))
        layout.addWidget(self._callout(
            "Use validation mAP together with visual review. A high mAP does not rule out systematic misses during full occlusion or wall contact."
        ))
        layout.addStretch(1); return self._scroll(page)

    def _hybrid_tab(self):
        page, layout = self._page()
        sr, form = self._group("Super-resolution crop stage")
        self._combo(form, "super_resolution.method", "Method", ["edsr", "espcn", "fsrcnn", "lapsrn", "bicubic", "none"], "edsr")
        self._line(form, "super_resolution.model_path", "OpenCV SR model (.pb)", browse="pb")
        self._spin(form, "super_resolution.scale", "Native SR scale", 4, 2, 8)
        self._spin(form, "super_resolution.output_size", "DLC crop video size", 512, 64, 2048)
        layout.addWidget(sr)

        dlc, dform = self._group("Pretrained DeepLabCut on enhanced crops")
        self._combo(dform, "model.superanimal_name", "SuperAnimal", ["superanimal_topviewmouse"], "superanimal_topviewmouse")
        self._combo(dform, "model.model_name", "Pose model", ["hrnet_w32", "resnet_50"], "hrnet_w32")
        self._combo(dform, "model.detector_name", "DLC detector", ["fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn"], "fasterrcnn_resnet50_fpn_v2")
        self._combo(dform, "model.device", "DLC device", ["auto", "cuda", "cpu", "mps"], "auto")
        self._spin(dform, "model.batch_size", "Pose batch size", 4, 1, 1024)
        self._spin(dform, "model.detector_batch_size", "Detector batch size", 1, 1, 1024)
        self._double(dform, "model.inference_pcutoff", "Raw DLC threshold", .10)
        self._check(dform, "model.video_adapt", "Self-supervised video adaptation", True)
        self._spin(dform, "model.video_adapt_batch_size", "Adaptation batch size", 4, 1, 1024)
        self._double(dform, "model.pseudo_threshold", "Pose pseudo-label threshold", .10)
        self._double(dform, "model.bbox_threshold", "DLC box pseudo-label threshold", .90)
        self._spin(dform, "model.detector_epochs", "Adaptation detector epochs", 1, 0, 1000)
        self._spin(dform, "model.pose_epochs", "Adaptation pose epochs", 1, 0, 1000)
        layout.addWidget(dlc)

        post, pform = self._group("Inverse mapping and final trajectory")
        self._double(pform, "postprocess.pcutoff", "Fusion confidence threshold", .35)
        self._double(pform, "postprocess.max_gap_sec", "Maximum interpolated gap (s)", .20, 0, 10)
        self._spin(pform, "postprocess.median_window", "Median filter window", 3, 1, 101)
        self._check(pform, "postprocess.write_overlay", "Write source-video overlay", True)
        self._check(pform, "postprocess.draw_all_keypoints", "Draw all reliable keypoints", True)
        layout.addWidget(post)
        layout.addWidget(self._actions([
            ("1 · YOLO + SR video", "prepare_hybrid"), ("2 · DLC on crops", "run_hybrid_dlc"),
            ("3 · Inverse-map + export", "postprocess_hybrid"), ("Run complete hybrid pipeline", "full_hybrid"),
        ], "Intermediate detection video, crop video, and per-frame transforms are retained for QA."))
        layout.addStretch(1); return self._scroll(page)

    def _dlc_training_tab(self):
        page, layout = self._page()
        project, form = self._group("Optional supervised DLC fine-tuning")
        self._line(form, "project.task", "Project name", "mouse_occlusion")
        self._line(form, "project.experimenter", "Experimenter", "researcher")
        self._check(form, "project.copy_videos", "Copy videos into project", False)
        self._double(form, "project.training_fraction", "Training fraction", .90, .1, .99, 2)
        self._spin(form, "project.num_frames", "Frames to extract", 40, 1, 100000)
        default_parts = "\n".join([
            "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip", "left_eye", "right_eye", "neck",
            "mid_back", "mouse_center", "mid_backend", "mid_backend2", "mid_backend3", "tail_base", "tail1", "tail2",
            "tail3", "tail4", "tail5", "left_shoulder", "left_midside", "left_hip", "right_shoulder", "right_midside",
            "right_hip", "tail_end", "head_midpoint",
        ])
        self._text(form, "project.bodyparts", "Bodyparts (one per line)", default_parts, 150)
        layout.addWidget(project)
        dataset, eform = self._group("DLC frames and labels")
        self._combo(eform, "dataset.mode", "Extraction mode", ["automatic", "manual"], "automatic")
        self._combo(eform, "dataset.algorithm", "Automatic algorithm", ["kmeans", "uniform"], "kmeans")
        self._check(eform, "dataset.crop", "Interactive crop before extraction", False)
        self._spin(eform, "dataset.cluster_step", "Cluster step", 1, 1, 10000)
        self._spin(eform, "dataset.cluster_resize_width", "Cluster resize width", 30, 10, 2000)
        self._check(eform, "dataset.cluster_color", "Cluster in color", False)
        layout.addWidget(dataset)
        training, tform = self._group("SuperAnimal transfer training")
        self._spin(tform, "training.shuffle", "Shuffle", 1, 1, 9999)
        self._combo(tform, "training.device", "Device", ["auto", "cuda", "cpu", "mps"], "auto")
        self._spin(tform, "training.epochs", "Pose epochs", 100, 1, 100000)
        self._spin(tform, "training.batch_size", "Pose batch size", 8, 1, 4096)
        self._spin(tform, "training.save_epochs", "Save every N epochs", 10, 1, 100000)
        self._spin(tform, "training.detector_epochs", "Detector epochs (0=freeze)", 0, 0, 100000)
        self._spin(tform, "training.detector_batch_size", "Detector batch size", 2, 1, 4096)
        self._spin(tform, "training.display_iters", "Log every N iterations", 20, 1, 100000)
        self._spin(tform, "training.max_snapshots", "Maximum snapshots", 5, 1, 1000)
        self._check(tform, "training.evaluation_plots", "Create evaluation plots", True)
        layout.addWidget(training)
        layout.addWidget(self._actions([
            ("Create DLC project", "create_project"), ("Add videos", "add_videos"),
            ("Extract DLC frames", "extract_frames"), ("Open DLC labeler", "label_frames"),
            ("Check labels", "check_labels"), ("Build transfer dataset", "build_dataset"),
            ("Train DLC", "train"), ("Evaluate DLC", "evaluate"),
        ]))
        layout.addStretch(1); return self._scroll(page)

    def _tuning_tab(self):
        page, layout = self._page()
        layout.addWidget(self._callout(
            "Failure-mining loop for the supervised DLC branch: analyze hard videos, extract outliers, correct labels, merge, rebuild, retrain, and compare evaluation metrics."
        ))
        box, form = self._group("DLC outlier selection")
        self._combo(form, "tuning.outlier_algorithm", "Outlier algorithm", ["jump", "uncertain", "fitting", "manual"], "jump")
        self._double(form, "tuning.epsilon", "Jump threshold epsilon (px)", 20, 0, 10000, 2)
        self._double(form, "tuning.p_bound", "Low-confidence bound", .10)
        self._combo(form, "tuning.extraction_algorithm", "Outlier deduplication", ["kmeans", "uniform"], "kmeans")
        self._check(form, "tuning.automatic", "Automatic frame selection", True)
        layout.addWidget(box)
        inference, iform = self._group("Fine-tuned DLC video inference")
        self._combo(iform, "inference.device", "Device", ["auto", "cuda", "cpu", "mps"], "auto")
        self._spin(iform, "inference.batch_size", "Pose batch size", 4, 1, 4096)
        self._spin(iform, "inference.detector_batch_size", "Detector batch size", 1, 1, 4096)
        self._double(iform, "inference.pcutoff", "Plot threshold", .35)
        self._check(iform, "inference.save_as_csv", "Write DLC CSV", True)
        self._check(iform, "inference.overwrite", "Overwrite predictions", False)
        self._check(iform, "inference.use_filtered", "Use filtered predictions", False)
        self._check(iform, "inference.draw_skeleton", "Draw skeleton", True)
        self._check(iform, "inference.plot_bboxes", "Draw detector boxes", True)
        self._spin(iform, "inference.trailpoints", "Trail length", 0, 0, 10000)
        layout.addWidget(inference)
        layout.addWidget(self._actions([
            ("Analyze", "analyze"), ("Filter predictions", "filter_predictions"),
            ("Create labeled video", "create_labeled_video"), ("Plot trajectories", "plot_trajectories"),
            ("Extract outliers", "extract_outliers"), ("Refine labels", "refine_labels"),
            ("Merge refined data", "merge_datasets"), ("Rebuild dataset", "build_dataset"),
            ("Retrain", "train"), ("Re-evaluate", "evaluate"),
        ]))
        layout.addWidget(self._callout(
            "Do not hide failures by continuously lowering confidence thresholds. Track YOLO misses, temporal fallbacks, DLC keypoint misses, and anatomical swaps separately.", True
        ))
        layout.addStretch(1); return self._scroll(page)

    def _advanced_tab(self):
        page, layout = self._page()
        layout.addWidget(self._callout(
            "Any parameter not shown in the forms can be passed here. Each top-level key is an action name; its value is merged into that Python API call and overrides the visible defaults."
        ))
        box, form = self._group("Advanced keyword arguments (JSON)")
        example = json.dumps({
            "train_yolo": {}, "validate_yolo": {}, "hybrid_dlc": {},
            "create_project": {}, "add_videos": {}, "extract_frames": {}, "label_frames": {}, "check_labels": {},
            "build_dataset": {}, "train": {"pytorch_cfg_updates": {}}, "evaluate": {},
            "analyze": {}, "filter_predictions": {}, "labeled_video": {}, "plot_trajectories": {},
            "extract_outliers": {}, "refine_labels": {}, "merge_datasets": {},
        }, indent=2)
        self._text(form, "advanced", "", example, 500)
        layout.addWidget(box)
        validate = QPushButton("Validate JSON"); validate.clicked.connect(self.validate_advanced); layout.addWidget(validate)
        layout.addStretch(1); return self._scroll(page)

    def _log_tab(self):
        page, layout = self._page()
        controls = QHBoxLayout()
        clear = QPushButton("Clear log"); clear.clicked.connect(lambda: self.log.clear())
        copy = QPushButton("Copy all"); copy.clicked.connect(lambda: QApplication.clipboard().setText(self.log.toPlainText()))
        output = QPushButton("Open output directory"); output.clicked.connect(self.open_output)
        controls.addWidget(clear); controls.addWidget(copy); controls.addWidget(output); controls.addStretch(1)
        layout.addLayout(controls)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas", 10)); layout.addWidget(self.log, 1)
        return page

    def _style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f7f9; color: #1e293b; font-size: 13px; }
            QLabel#title { font-size: 25px; font-weight: 700; color: #123b5d; }
            QLabel#subtitle, QLabel#configPath, QLabel#hint { color: #64748b; }
            QLabel#callout { background: #e7f3fa; border-left: 4px solid #1683b0; padding: 12px; border-radius: 4px; }
            QLabel#warning { background: #fff3df; border-left: 4px solid #d97706; padding: 12px; border-radius: 4px; }
            QLabel#status { font-weight: 600; color: #245a73; }
            QTabWidget::pane { border: 1px solid #d5dee5; background: white; border-radius: 5px; }
            QTabBar::tab { padding: 10px 15px; background: #e6edf2; margin-right: 2px; }
            QTabBar::tab:selected { background: white; color: #126488; font-weight: 600; }
            QGroupBox { background: white; border: 1px solid #d5dee5; border-radius: 6px; margin-top: 13px; padding: 14px 10px 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #285b76; }
            QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: white; border: 1px solid #c7d2dc; border-radius: 4px; padding: 5px; selection-background-color: #3b8eb6; }
            QPushButton { background: #e6edf2; border: 1px solid #c3ced7; border-radius: 4px; padding: 7px 12px; }
            QPushButton:hover { background: #d9e7ee; }
            QPushButton[primary="true"] { background: #1679a5; color: white; border-color: #126488; font-weight: 600; }
            QPushButton[primary="true"]:hover { background: #126488; }
            QPushButton:disabled { color: #94a3b8; background: #edf1f4; }
            QProgressBar { border: 1px solid #c7d2dc; border-radius: 4px; height: 10px; background: white; }
            QProgressBar::chunk { background: #2789b4; }
        """)

    def browse(self, field: QLineEdit, mode: str) -> None:
        current = field.text().strip() or str(HERE)
        if mode == "dir": value = QFileDialog.getExistingDirectory(self, "Select directory", current)
        else:
            filters = {
                "json": "JSON (*.json)", "yaml": "YAML (*.yaml *.yml)", "model": "PyTorch model (*.pt)",
                "model_optional": "PyTorch model (*.pt);;All files (*)", "pb": "OpenCV super-resolution model (*.pb)",
            }.get(mode, "All files (*)")
            value, _ = QFileDialog.getOpenFileName(self, "Select file", current, filters)
        if value: field.setText(value)

    def choose_config(self) -> None:
        value, _ = QFileDialog.getOpenFileName(self, "Load workbench configuration", str(self.config_path.parent), "JSON (*.json)")
        if value: self.load_config(Path(value))

    def load_config(self, path: Path) -> None:
        try: data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", str(exc)); return
        self.config_path = DEFAULT_CONFIG if path.name == EXAMPLE_CONFIG.name else path.resolve()
        for key, widget in self.fields.items():
            if key == "advanced": value = data.get("advanced")
            elif key == "videos": value = data.get("videos") or ([data["video"]] if data.get("video") else None)
            else: value = nested_get(data, key)
            if value is None: continue
            if isinstance(widget, QLineEdit): widget.setText(str(value))
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(json.dumps(value, indent=2) if key == "advanced" else ("\n".join(map(str, value)) if isinstance(value, list) else str(value)))
            elif isinstance(widget, QCheckBox): widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                if widget.findText(str(value)) < 0: widget.addItem(str(value))
                widget.setCurrentText(str(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)): widget.setValue(value)
        self.config_label.setText(str(self.config_path)); self.status.setText("Configuration loaded")

    def collect(self) -> dict:
        data: dict[str, Any] = {}
        for key, widget in self.fields.items():
            if isinstance(widget, QLineEdit): value: Any = widget.text().strip()
            elif isinstance(widget, QPlainTextEdit):
                text = widget.toPlainText().strip()
                value = json.loads(text or "{}") if key == "advanced" else [line.strip() for line in text.splitlines() if line.strip()]
            elif isinstance(widget, QCheckBox): value = widget.isChecked()
            elif isinstance(widget, QComboBox): value = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)): value = widget.value()
            else: continue
            if key == "advanced": data["advanced"] = value
            elif key == "videos":
                data["videos"] = value
                if value: data["video"] = value[0]
            else: nested_set(data, key, value)
        return data

    def save_config(self) -> bool:
        try:
            data = self.collect(); self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Cannot save configuration", str(exc)); return False
        self.config_label.setText(str(self.config_path)); self.status.setText("Configuration saved"); return True

    def save_as(self) -> None:
        value, _ = QFileDialog.getSaveFileName(self, "Save configuration", str(self.config_path), "JSON (*.json)")
        if value: self.config_path = Path(value); self.save_config()

    def validate_advanced(self) -> None:
        try:
            value = json.loads(self.fields["advanced"].toPlainText() or "{}")
            if not isinstance(value, dict) or not all(isinstance(item, dict) for item in value.values()):
                raise ValueError("Every top-level action must contain a JSON object")
        except Exception as exc: QMessageBox.warning(self, "Invalid JSON", str(exc)); return
        QMessageBox.information(self, "Advanced API", "The JSON structure is valid.")

    def open_box_review(self) -> None:
        if not self.save_config(): return
        try:
            import dlc.box_review as review
            # box_review intentionally avoids importing math until the dialog is opened.
            review.math = math
            dataset_field = self.fields["yolo.dataset_dir"]
            assert isinstance(dataset_field, QLineEdit)
            dataset = Path(dataset_field.text()).expanduser()
            if not dataset.is_absolute(): dataset = (self.config_path.parent / dataset).resolve()
            review.open_box_review(dataset / "box_labels.csv", self)
        except Exception as exc: QMessageBox.critical(self, "Box review error", str(exc))

    def start_job(self, action: str, label: str) -> None:
        if self.process is not None:
            QMessageBox.warning(self, "Task already running", "Stop or wait for the current task."); return
        if not self.save_config(): return
        self.tabs.setCurrentIndex(self.tabs.count() - 1)
        command = [str(HERE / "hybrid_jobs.py"), action, "--config", str(self.config_path)]
        self.log.append(f"\n$ {sys.executable} {' '.join(command)}\n")
        process = QProcess(self); process.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment(); env.insert("PYTHONUNBUFFERED", "1"); process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self.read_output); process.finished.connect(self.job_finished); process.errorOccurred.connect(self.job_error)
        self.process = process; self.status.setText(f"Running: {label}"); self.progress.setRange(0, 0); self.stop_button.setEnabled(True)
        process.start(sys.executable, command)

    def read_output(self) -> None:
        if self.process is None: return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log.moveCursor(QTextCursor.End); self.log.insertPlainText(text); self.log.ensureCursorVisible()
        for line in text.splitlines():
            if not line.startswith("HYBRID_GUI_RESULT "): continue
            try:
                result = json.loads(line.removeprefix("HYBRID_GUI_RESULT "))
                if result.get("yolo_trained_model"):
                    field = self.fields["yolo.trained_model"]; assert isinstance(field, QLineEdit); field.setText(result["yolo_trained_model"]); self.save_config()
            except Exception: pass

    def job_finished(self, code: int, _status) -> None:
        self.progress.setRange(0, 1); self.progress.setValue(1 if code == 0 else 0); self.stop_button.setEnabled(False)
        self.status.setText("Task completed" if code == 0 else f"Task failed (exit code {code}); inspect Live Log")
        self.process = None

    def job_error(self, error) -> None:
        self.log.append(f"\nProcess error: {error}\n")
        if self.process is not None and self.process.state() == QProcess.NotRunning:
            self.process = None; self.progress.setRange(0, 1); self.stop_button.setEnabled(False); self.status.setText("Process failed to start")

    def stop_job(self) -> None:
        if self.process is None: return
        self.process.terminate()
        if not self.process.waitForFinished(3000): self.process.kill()
        self.status.setText("Task stopped; verify the last checkpoint before resuming")

    def open_output(self) -> None:
        field = self.fields["output_dir"]; assert isinstance(field, QLineEdit)
        path = Path(field.text()).expanduser()
        if not path.is_absolute(): path = (self.config_path.parent / path).resolve()
        path.mkdir(parents=True, exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def main() -> int:
    app = QApplication(sys.argv); app.setApplicationName("Mouse Pose Hybrid Workbench")
    window = HybridWorkbench(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
