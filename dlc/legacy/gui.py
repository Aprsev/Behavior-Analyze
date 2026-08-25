#!/usr/bin/env python3
"""DeepLabCut mouse tracking workbench (PySide6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
    from PySide6.QtGui import QDesktopServices, QFont
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
        QPushButton, QScrollArea, QSpinBox, QTabWidget, QTextEdit, QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - shown only in incomplete envs
    raise SystemExit("PySide6 is missing. Install dlc/requirements-dlc.txt first.") from exc

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
EXAMPLE_CONFIG = HERE / "config.example.json"


def nested_get(data: dict, dotted: str, default: Any = None) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def nested_set(data: dict, dotted: str, value: Any) -> None:
    target = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


class Workbench(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeepLabCut 小鼠遮挡关键点工作台")
        self.resize(1280, 850)
        self.fields: dict[str, QWidget] = {}
        self.process: QProcess | None = None
        self.config_path = DEFAULT_CONFIG
        self._build_ui()
        self._apply_style()
        self.load_config(DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else EXAMPLE_CONFIG)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("DeepLabCut Mouse Workbench")
        title.setObjectName("title")
        subtitle = QLabel("预训练测试 · 数据标注 · SuperAnimal 迁移训练 · 批量推理 · 遮挡调优")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.config_label = QLabel()
        self.config_label.setObjectName("pathLabel")
        header.addWidget(self.config_label)
        for text, callback in (("载入配置", self.choose_config), ("保存配置", self.save_config), ("另存为", self.save_config_as)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            header.addWidget(button)
        outer.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._overview_tab(), "① 环境与项目")
        self.tabs.addTab(self._pretrained_tab(), "② 预训练测试")
        self.tabs.addTab(self._dataset_tab(), "③ 数据与标注")
        self.tabs.addTab(self._training_tab(), "④ 训练与评估")
        self.tabs.addTab(self._inference_tab(), "⑤ 测试与可视化")
        self.tabs.addTab(self._tuning_tab(), "⑥ 模型调优")
        self.tabs.addTab(self._advanced_tab(), "高级参数")
        self.tabs.addTab(self._log_tab(), "运行日志")
        outer.addWidget(self.tabs, 1)

        status_bar = QHBoxLayout()
        self.status = QLabel("就绪")
        self.status.setObjectName("status")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.cancel_button = QPushButton("停止当前任务")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_job)
        status_bar.addWidget(self.status, 1)
        status_bar.addWidget(self.progress, 2)
        status_bar.addWidget(self.cancel_button)
        outer.addLayout(status_bar)
        self.setCentralWidget(root)

    def _scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(content)
        return area

    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(12)
        return page, layout

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return box, form

    def _line(self, form: QFormLayout, key: str, label: str, default: str = "", browse: str | None = None) -> QLineEdit:
        widget = QLineEdit(default)
        self.fields[key] = widget
        if browse:
            row = QHBoxLayout()
            row.addWidget(widget, 1)
            button = QPushButton("浏览…")
            button.clicked.connect(lambda _=False, w=widget, mode=browse: self.browse_path(w, mode))
            row.addWidget(button)
            form.addRow(label, row)
        else:
            form.addRow(label, widget)
        return widget


    def _text(self, form: QFormLayout, key: str, label: str, default: str = "", height: int = 90) -> QPlainTextEdit:
        widget = QPlainTextEdit(default)
        widget.setMinimumHeight(height)
        self.fields[key] = widget
        form.addRow(label, widget)
        return widget


    def _spin(self, form: QFormLayout, key: str, label: str, default: int, low: int = 0, high: int = 1_000_000) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(low, high)
        widget.setValue(default)
        self.fields[key] = widget
        form.addRow(label, widget)
        return widget


    def _double(self, form: QFormLayout, key: str, label: str, default: float, low: float = 0, high: float = 1, decimals: int = 3) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(decimals)
        widget.setSingleStep(0.05)
        widget.setValue(default)
        self.fields[key] = widget
        form.addRow(label, widget)
        return widget


    def _check(self, form: QFormLayout, key: str, label: str, default: bool) -> QCheckBox:
        widget = QCheckBox()
        widget.setChecked(default)
        self.fields[key] = widget
        form.addRow(label, widget)
        return widget


    def _combo(self, form: QFormLayout, key: str, label: str, values: list[str], default: str) -> QComboBox:
        widget = QComboBox()
        widget.addItems(values)
        if default in values:
            widget.setCurrentText(default)
        self.fields[key] = widget
        form.addRow(label, widget)
        return widget


    def _buttons(self, specs: list[tuple[str, str]], note: str = "") -> QWidget:
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 4)
        for text, action in specs:
            button = QPushButton(text)
            button.setProperty("primary", action in {"zero_shot", "create_project", "build_dataset", "train", "analyze"})
            button.clicked.connect(lambda _=False, a=action, t=text: self.start_job(a, t))
            layout.addWidget(button)
        layout.addStretch(1)
        if note:
            label = QLabel(note)
            label.setWordWrap(True)
            label.setObjectName("hint")
            layout.addWidget(label, 2)
        return box


    def _overview_tab(self) -> QScrollArea:
        page, layout = self._page()
        intro = QLabel("从上到下完成六个阶段。所有长任务都在独立进程运行；关闭标注窗口后，工作台会自动收到结束状态。")
        intro.setWordWrap(True)
        intro.setObjectName("callout")
        layout.addWidget(intro)
        box, form = self._group("公共路径与实验几何")
        self._text(form, "videos", "输入视频（每行一个）", height=105)
        self._line(form, "roi_json", "四角 ROI JSON", browse="file_json")
        self._line(form, "output_dir", "结果目录", browse="dir")
        self._double(form, "arena_width_cm", "场地宽度 (cm)", 25, 0.1, 1000, 2)
        self._double(form, "arena_height_cm", "场地高度 (cm)", 30, 0.1, 1000, 2)
        self._line(form, "project_config", "DLC config.yaml", browse="file_yaml")
        self._line(form, "working_directory", "DLC 项目根目录", browse="dir")
        layout.addWidget(box)
        layout.addWidget(self._buttons([("检查 Python / CUDA / 文件", "check")], "检查不会修改任何数据。首次使用建议先运行。"))
        steps = QLabel("推荐顺序：预训练短视频测试 → 创建项目 → 提取并标注多样帧 → 构建迁移训练集 → 训练与评估 → 新视频测试 → 回收失败帧继续调优。")
        steps.setWordWrap(True)
        steps.setObjectName("callout")
        layout.addWidget(steps)
        layout.addStretch(1)
        return self._scroll(page)


    def _pretrained_tab(self) -> QScrollArea:
        page, layout = self._page()
        box, form = self._group("SuperAnimal-TopViewMouse 零样本 / 自监督适配")
        self._combo(form, "model.superanimal_name", "预训练体系", ["superanimal_topviewmouse"], "superanimal_topviewmouse")
        self._combo(form, "model.model_name", "姿态网络", ["hrnet_w32", "resnet_50"], "hrnet_w32")
        self._combo(form, "model.detector_name", "检测器", ["fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn"], "fasterrcnn_resnet50_fpn_v2")
        self._combo(form, "model.device", "设备", ["auto", "cuda", "cpu", "mps"], "auto")
        self._spin(form, "model.max_individuals", "最大动物数", 1, 1, 30)
        self._spin(form, "model.batch_size", "姿态 batch size", 4, 1, 1024)
        self._spin(form, "model.detector_batch_size", "检测 batch size", 1, 1, 1024)
        self._double(form, "model.inference_pcutoff", "DLC 原始输出阈值", 0.10)
        self._check(form, "model.video_adapt", "启用无标注视频适配", True)
        self._spin(form, "model.video_adapt_batch_size", "适配 batch size", 4, 1, 1024)
        self._double(form, "model.pseudo_threshold", "姿态伪标签阈值", 0.10)
        self._double(form, "model.bbox_threshold", "检测框伪标签阈值", 0.90)
        self._spin(form, "model.detector_epochs", "适配检测器 epochs", 1, 0, 1000)
        self._spin(form, "model.pose_epochs", "适配姿态 epochs", 1, 0, 1000)
        self._check(form, "model.plot_bboxes", "绘制检测框", True)
        self._check(form, "model.create_labeled_video", "生成 DLC 标注视频", True)
        layout.addWidget(box)

        post, pform = self._group("遮挡融合与项目格式导出")
        self._double(pform, "postprocess.pcutoff", "关键点融合阈值", 0.35)
        self._double(pform, "postprocess.max_gap_sec", "最大短缺口 (秒)", 0.20, 0, 10, 3)
        self._spin(pform, "postprocess.median_window", "中值滤波窗口", 3, 1, 101)
        self._check(pform, "postprocess.write_overlay", "生成融合标注视频", True)
        self._check(pform, "postprocess.draw_all_keypoints", "绘制全部可靠关键点", True)
        layout.addWidget(post)
        layout.addWidget(self._buttons([("运行预训练推理", "zero_shot"), ("仅重新后处理", "postprocess")], "完全遮挡超过设定时长时保留 NaN。"))
        layout.addStretch(1)
        return self._scroll(page)


    def _dataset_tab(self) -> QScrollArea:
        page, layout = self._page()
        project, form = self._group("创建单动物 PyTorch 项目")
        self._line(form, "project.task", "项目名", "mouse_occlusion")
        self._line(form, "project.experimenter", "实验者", "researcher")
        self._check(form, "project.copy_videos", "复制视频到项目", False)
        self._double(form, "project.training_fraction", "训练集比例", 0.90, 0.1, 0.99, 2)
        self._spin(form, "project.num_frames", "每项目抽取帧数", 40, 1, 100000)
        default_parts = "\n".join([
            "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip", "left_eye", "right_eye", "neck",
            "mid_back", "mouse_center", "mid_backend", "mid_backend2", "mid_backend3", "tail_base", "tail1", "tail2",
            "tail3", "tail4", "tail5", "left_shoulder", "left_midside", "left_hip", "right_shoulder", "right_midside",
            "right_hip", "tail_end", "head_midpoint",
        ])
        self._text(form, "project.bodyparts", "关键点（每行一个）", default_parts, 170)
        layout.addWidget(project)
        layout.addWidget(self._buttons([("创建项目", "create_project"), ("向现有项目添加视频", "add_videos")]))

        extract, eform = self._group("关键帧提取与标注")
        self._combo(eform, "dataset.mode", "提取模式", ["automatic", "manual"], "automatic")
        self._combo(eform, "dataset.algorithm", "自动算法", ["kmeans", "uniform"], "kmeans")
        self._check(eform, "dataset.crop", "提取前交互裁剪", False)
        self._spin(eform, "dataset.cluster_step", "聚类抽帧步长", 1, 1, 10000)
        self._spin(eform, "dataset.cluster_resize_width", "聚类缩放宽度", 30, 10, 2000)
        self._check(eform, "dataset.cluster_color", "使用颜色聚类", False)
        layout.addWidget(extract)
        layout.addWidget(self._buttons([
            ("提取关键帧", "extract_frames"), ("打开 DLC 标注器", "label_frames"), ("生成标签检查图", "check_labels")
        ], "障碍边缘、完全/部分遮挡、靠墙、转身、静止和不同动物都应进入标注集。"))
        layout.addStretch(1)
        return self._scroll(page)


    def _training_tab(self) -> QScrollArea:
        page, layout = self._page()
        box, form = self._group("SuperAnimal 权重迁移训练")
        self._spin(form, "training.shuffle", "Shuffle", 1, 1, 9999)
        self._combo(form, "training.device", "训练设备", ["auto", "cuda", "cpu", "mps"], "auto")
        self._spin(form, "training.epochs", "姿态网络 epochs", 100, 1, 100000)
        self._spin(form, "training.batch_size", "姿态 batch size", 8, 1, 4096)
        self._spin(form, "training.save_epochs", "每 N epochs 保存", 10, 1, 100000)
        self._spin(form, "training.detector_epochs", "检测器 epochs（0=冻结）", 0, 0, 100000)
        self._spin(form, "training.detector_batch_size", "检测器 batch size", 2, 1, 4096)
        self._spin(form, "training.display_iters", "每 N iterations 输出", 20, 1, 100000)
        self._spin(form, "training.max_snapshots", "最多保留快照", 5, 1, 1000)
        self._check(form, "training.evaluation_plots", "评估时绘图", True)
        layout.addWidget(box)
        layout.addWidget(self._buttons([
            ("构建迁移训练数据集", "build_dataset"), ("开始 / 继续训练", "train"), ("评估模型", "evaluate")
        ], "先完成标注并检查标签，再构建训练集。显存不足时先减小两个 batch size。"))
        tip = QLabel("检测器默认冻结（detector epochs=0），先迁移训练关键点网络。若障碍物导致整只小鼠检测框经常丢失，再加入对应帧并训练检测器。")
        tip.setWordWrap(True)
        tip.setObjectName("callout")
        layout.addWidget(tip)
        layout.addStretch(1)
        return self._scroll(page)


    def _inference_tab(self) -> QScrollArea:
        page, layout = self._page()
        box, form = self._group("已训练模型的批量测试")
        self._combo(form, "inference.device", "推理设备", ["auto", "cuda", "cpu", "mps"], "auto")
        self._spin(form, "inference.batch_size", "姿态 batch size", 4, 1, 4096)
        self._spin(form, "inference.detector_batch_size", "检测 batch size", 1, 1, 4096)
        self._double(form, "inference.pcutoff", "可视化置信度阈值", 0.35)
        self._check(form, "inference.save_as_csv", "同时保存 DLC CSV", True)
        self._check(form, "inference.overwrite", "覆盖已有预测", False)
        self._check(form, "inference.use_filtered", "视频/图表使用 DLC 滤波结果", False)
        self._check(form, "inference.draw_skeleton", "绘制骨架", True)
        self._check(form, "inference.plot_bboxes", "绘制检测框", True)
        self._spin(form, "inference.trailpoints", "轨迹尾迹帧数", 0, 0, 10000)
        layout.addWidget(box)
        layout.addWidget(self._buttons([
            ("① 批量分析视频", "analyze"), ("② DLC 滤波", "filter_predictions"),
            ("③ 生成标注视频", "create_labeled_video"), ("轨迹诊断图", "plot_trajectories"),
            ("导出本项目头部/质心", "postprocess"),
        ]))
        output_row = QHBoxLayout()
        open_output = QPushButton("打开结果目录")
        open_output.clicked.connect(self.open_output)
        open_project = QPushButton("打开 DLC config.yaml")
        open_project.clicked.connect(self.open_project_config)
        output_row.addWidget(open_output)
        output_row.addWidget(open_project)
        output_row.addStretch(1)
        layout.addLayout(output_row)
        layout.addStretch(1)
        return self._scroll(page)


    def _tuning_tab(self) -> QScrollArea:
        page, layout = self._page()
        callout = QLabel("闭环调优：先用当前模型分析困难视频，再自动找异常帧，人工修正，合并数据集，重新构建并训练。每轮都保留 DLC iteration 和评估结果。")
        callout.setWordWrap(True)
        callout.setObjectName("callout")
        layout.addWidget(callout)
        box, form = self._group("失败帧挖掘")
        self._combo(form, "tuning.outlier_algorithm", "异常判定", ["jump", "uncertain", "fitting", "manual"], "jump")
        self._double(form, "tuning.epsilon", "跳跃阈值 epsilon (px)", 20, 0, 10000, 2)
        self._double(form, "tuning.p_bound", "低置信度界限", 0.10)
        self._combo(form, "tuning.extraction_algorithm", "异常帧去重", ["kmeans", "uniform"], "kmeans")
        self._check(form, "tuning.automatic", "自动执行帧选择", True)
        layout.addWidget(box)
        layout.addWidget(self._buttons([
            ("① 用当前模型分析", "analyze"), ("② 提取异常帧", "extract_outliers"),
            ("③ 打开修正标注器", "refine_labels"), ("④ 合并修正数据", "merge_datasets"),
            ("⑤ 重建迁移训练集", "build_dataset"), ("⑥ 再训练", "train"), ("⑦ 再评估", "evaluate"),
        ]))
        warning = QLabel("不要通过持续降低 pcutoff 掩盖错误。完全遮挡、检测框丢失和关键点错位应分别统计，并把代表性失败帧加入训练集。")
        warning.setWordWrap(True)
        warning.setObjectName("warning")
        layout.addWidget(warning)
        layout.addStretch(1)
        return self._scroll(page)


    def _advanced_tab(self) -> QScrollArea:
        page, layout = self._page()
        help_text = QLabel(
            "这里可以覆盖每个 DeepLabCut API 的任意命名参数，因此 GUI 不会限制高级配置。"
            "顶层键使用动作名，例如 train、analyze、zero_shot；值必须是传给该 API 的 JSON 对象。"
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("callout")
        layout.addWidget(help_text)
        box, form = self._group("高级 kwargs（JSON）")
        example = json.dumps({
            "zero_shot": {},
            "create_project": {}, "add_videos": {}, "extract_frames": {}, "label_frames": {}, "check_labels": {},
            "build_dataset": {}, "train": {"pytorch_cfg_updates": {}}, "evaluate": {},
            "analyze": {}, "filter_predictions": {}, "labeled_video": {}, "plot_trajectories": {},
            "extract_outliers": {}, "refine_labels": {}, "merge_datasets": {},
        }, indent=2, ensure_ascii=False)
        self._text(form, "advanced", "", example, 480)
        layout.addWidget(box)
        validate = QPushButton("验证高级 JSON")
        validate.clicked.connect(self.validate_advanced)
        layout.addWidget(validate)
        layout.addStretch(1)
        return self._scroll(page)


    def _log_tab(self) -> QWidget:
        page, layout = self._page()
        row = QHBoxLayout()
        clear = QPushButton("清空日志")
        clear.clicked.connect(lambda: self.log.clear())
        copy = QPushButton("复制全部")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.log.toPlainText()))
        row.addWidget(clear)
        row.addWidget(copy)
        row.addStretch(1)
        layout.addLayout(row)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log, 1)
        return page


    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f5f7fa; color: #1f2937; font-size: 13px; }
            QLabel#title { font-size: 24px; font-weight: 700; color: #123b5d; }
            QLabel#subtitle { color: #64748b; padding-bottom: 6px; }
            QLabel#pathLabel { color: #64748b; max-width: 300px; }
            QLabel#callout { background: #e8f3fb; border-left: 4px solid #2384b8; padding: 12px; border-radius: 4px; }
            QLabel#warning { background: #fff4e5; border-left: 4px solid #d97706; padding: 12px; border-radius: 4px; }
            QLabel#hint { color: #64748b; }
            QLabel#status { font-weight: 600; color: #285b76; }
            QTabWidget::pane { border: 1px solid #d8e0e7; background: white; border-radius: 5px; }
            QTabBar::tab { padding: 10px 15px; background: #e8edf2; margin-right: 2px; }
            QTabBar::tab:selected { background: white; color: #12618a; font-weight: 600; }
            QGroupBox { background: white; border: 1px solid #d8e0e7; border-radius: 6px; margin-top: 13px; padding: 14px 10px 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #285b76; }
            QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: white; border: 1px solid #cbd5df; border-radius: 4px; padding: 5px; selection-background-color: #3a8db8; }
            QPushButton { background: #e8edf2; border: 1px solid #c6d0d9; border-radius: 4px; padding: 7px 12px; }
            QPushButton:hover { background: #dce8ef; }
            QPushButton[primary="true"] { background: #1677a5; color: white; border-color: #12618a; font-weight: 600; }
            QPushButton[primary="true"]:hover { background: #12618a; }
            QPushButton:disabled { color: #94a3b8; background: #edf1f4; }
            QProgressBar { border: 1px solid #cbd5df; border-radius: 4px; height: 10px; background: white; }
            QProgressBar::chunk { background: #2b8ab7; }
        """)


    def browse_path(self, widget: QLineEdit, mode: str) -> None:
        current = widget.text().strip() or str(HERE)
        if mode == "dir":
            value = QFileDialog.getExistingDirectory(self, "选择目录", current)
        else:
            filters = "JSON (*.json)" if mode == "file_json" else "DeepLabCut config (config.yaml *.yml *.yaml)"
            value, _ = QFileDialog.getOpenFileName(self, "选择文件", current, filters)
        if value:
            widget.setText(value)


    def choose_config(self) -> None:
        value, _ = QFileDialog.getOpenFileName(self, "载入工作台配置", str(self.config_path.parent), "JSON (*.json)")
        if value:
            self.load_config(Path(value))


    def load_config(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "配置错误", str(exc))
            return
        self.config_path = path.resolve() if path.name != "config.example.json" else DEFAULT_CONFIG
        for key, widget in self.fields.items():
            if key == "advanced":
                value = data.get("advanced", {})
            elif key == "videos":
                value = data.get("videos") or ([data["video"]] if data.get("video") else [])
            else:
                value = nested_get(data, key, None)
            if value is None:
                continue
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QPlainTextEdit):
                if key == "advanced":
                    widget.setPlainText(json.dumps(value, indent=2, ensure_ascii=False))
                elif isinstance(value, list):
                    widget.setPlainText("\n".join(map(str, value)))
                else:
                    widget.setPlainText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                if widget.findText(str(value)) < 0:
                    widget.addItem(str(value))
                widget.setCurrentText(str(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
        self.config_label.setText(str(self.config_path))
        self.status.setText("配置已载入")


    def collect_config(self) -> dict:
        data: dict[str, Any] = {}
        for key, widget in self.fields.items():
            if isinstance(widget, QLineEdit):
                value: Any = widget.text().strip()
            elif isinstance(widget, QPlainTextEdit):
                text = widget.toPlainText().strip()
                if key == "advanced":
                    value = json.loads(text or "{}")
                else:
                    value = [line.strip() for line in text.splitlines() if line.strip()]
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            else:
                continue
            if key == "advanced":
                data["advanced"] = value
            elif key == "videos":
                data["videos"] = value
                if value:
                    data["video"] = value[0]
            else:
                nested_set(data, key, value)
        return data


    def save_config(self) -> bool:
        try:
            data = self.collect_config()
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "无法保存配置", str(exc))
            return False
        self.config_label.setText(str(self.config_path))
        self.status.setText("配置已保存")
        return True


    def save_config_as(self) -> None:
        value, _ = QFileDialog.getSaveFileName(self, "保存工作台配置", str(self.config_path), "JSON (*.json)")
        if value:
            self.config_path = Path(value)
            self.save_config()


    def validate_advanced(self) -> None:
        try:
            value = json.loads(self.fields["advanced"].toPlainText() or "{}")  # type: ignore[attr-defined]
            if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
                raise ValueError("顶层及每个动作的值都必须是 JSON object")
        except Exception as exc:
            QMessageBox.warning(self, "高级参数无效", str(exc))
            return
        QMessageBox.information(self, "高级参数", "JSON 结构有效。")


    def start_job(self, action: str, label: str) -> None:
        if self.process is not None:
            QMessageBox.warning(self, "任务正在运行", "请等待当前任务结束或先停止它。")
            return
        if not self.save_config():
            return
        self.tabs.setCurrentIndex(self.tabs.count() - 1)
        self.log.append(f"\n$ {sys.executable} {HERE / 'jobs.py'} {action} --config {self.config_path}\n")
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self.read_output)
        process.finished.connect(self.job_finished)
        process.errorOccurred.connect(self.job_error)
        self.process = process
        self.status.setText(f"正在执行：{label}")
        self.progress.setRange(0, 0)
        self.cancel_button.setEnabled(True)
        process.start(sys.executable, [str(HERE / "jobs.py"), action, "--config", str(self.config_path)])


    def read_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()
        for line in text.splitlines():
            if line.startswith("DLC_GUI_RESULT "):
                try:
                    result = json.loads(line.removeprefix("DLC_GUI_RESULT "))
                    if result.get("project_config"):
                        field = self.fields["project_config"]
                        assert isinstance(field, QLineEdit)
                        field.setText(result["project_config"])
                        self.save_config()
                except Exception:
                    pass


    def job_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if exit_code == 0 else 0)
        self.cancel_button.setEnabled(False)
        self.status.setText("任务完成" if exit_code == 0 else f"任务失败（退出码 {exit_code}），请查看日志")
        self.process = None


    def job_error(self, error: QProcess.ProcessError) -> None:
        self.log.append(f"\nQProcess error: {error}\n")


    def cancel_job(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self.status.setText("任务已停止；训练任务请检查最后一个快照是否完整")


    def open_output(self) -> None:
        field = self.fields["output_dir"]
        assert isinstance(field, QLineEdit)
        path = Path(field.text()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))


    def open_project_config(self) -> None:
        field = self.fields["project_config"]
        assert isinstance(field, QLineEdit)
        path = Path(field.text()).expanduser()
        if path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            QMessageBox.warning(self, "文件不存在", str(path))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("DeepLabCut Mouse Workbench")
    window = Workbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
