#!/usr/bin/env python3
"""PySide6 GUI for import -> annotate -> train -> analyse -> inspect.

All heavy work runs in a child Python process. Qt's GUI thread never reads a
video, trains a model, or writes an MP4. Output is streamed through QProcess
and displayed with a throttled timer, so long videos do not freeze the UI.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QPlainTextEdit, QProgressBar, QSpinBox, QDoubleSpinBox,
    QVBoxLayout, QWidget,
)


ROOT = Path(__file__).resolve().parents[2]  # traditional/
PROJECT_ROOT = ROOT.parent                 # repository root; data/ stays here
CODE = ROOT / "code"


@dataclass
class ProjectState:
    video: str = ""
    output_dir: str = ""
    roi_json: str = ""
    head_labels: str = ""
    torso_labels: str = ""
    model: str = ""
    arena_width_cm: float = 30.0
    arena_height_cm: float = 25.0


class WorkflowWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = ProjectState()
        self.process: QProcess | None = None
        self.pending_log: list[str] = []
        self.setWindowTitle("Behavior Analyze | Scientific Workflow")
        self.resize(1180, 780)
        self._build_ui()
        self.log_timer = QTimer(self); self.log_timer.timeout.connect(self._flush_log); self.log_timer.start(100)

    def _build_ui(self) -> None:
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central); outer.setContentsMargins(22, 18, 22, 18); outer.setSpacing(12)
        title = QLabel("Mouse Behavior Analysis")
        title.setObjectName("title"); outer.addWidget(title)
        subtitle = QLabel("Sequential scientific workflow: import → calibrate → label → train → infer → inspect")
        subtitle.setObjectName("subtitle"); outer.addWidget(subtitle)
        body = QHBoxLayout(); outer.addLayout(body, 1)
        left = QVBoxLayout(); left.setSpacing(10); body.addLayout(left, 3)
        right = QVBoxLayout(); body.addLayout(right, 2)

        self.video_edit = QLineEdit(); self.output_edit = QLineEdit(); self.roi_edit = QLineEdit(); self.model_edit = QLineEdit()
        self.head_labels_edit = QLineEdit(); self.torso_labels_edit = QLineEdit()
        self.width_spin = QDoubleSpinBox(); self.height_spin = QDoubleSpinBox()
        for spin, value in ((self.width_spin, 30), (self.height_spin, 25)):
            spin.setRange(1, 200); spin.setValue(value); spin.setSuffix(" cm"); spin.setDecimals(2)
        left.addWidget(self._import_group())
        left.addWidget(self._calibration_group())
        left.addWidget(self._annotation_group())
        left.addWidget(self._training_group())
        left.addWidget(self._inference_group())

        status_group = QGroupBox("Execution status")
        sl = QVBoxLayout(status_group)
        self.status = QLabel("Idle — choose a video to begin."); self.status.setObjectName("status")
        self.progress = QProgressBar(); self.progress.setRange(0, 1); self.progress.setValue(0)
        sl.addWidget(self.status); sl.addWidget(self.progress); right.addWidget(status_group)
        results = QGroupBox("Results")
        rl = QVBoxLayout(results)
        self.open_output = QPushButton("Open output folder"); self.open_output.clicked.connect(self._open_output)
        self.open_video = QPushButton("Open annotated result video"); self.open_video.clicked.connect(self._open_result_video)
        self.save_project = QPushButton("Save project state"); self.save_project.clicked.connect(self._save_project)
        rl.addWidget(self.open_output); rl.addWidget(self.open_video); rl.addWidget(self.save_project); right.addWidget(results)
        log_group = QGroupBox("Process log")
        ll = QVBoxLayout(log_group); self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(3000); ll.addWidget(self.log)
        right.addWidget(log_group, 1)
        self._style()

    def _path_row(self, edit: QLineEdit, choose) -> QWidget:
        row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1); button = QPushButton("Browse"); button.clicked.connect(choose); layout.addWidget(button); return row

    def _import_group(self) -> QGroupBox:
        group = QGroupBox("1. Import video and define project") ; form = QFormLayout(group)
        form.addRow("Video (.avi/.mp4)", self._path_row(self.video_edit, self._choose_video))
        form.addRow("Output directory", self._path_row(self.output_edit, self._choose_output))
        form.addRow("Arena width", self.width_spin); form.addRow("Arena height", self.height_spin)
        return group

    def _calibration_group(self) -> QGroupBox:
        group = QGroupBox("2. Arena calibration and body identity") ; layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Create automatic candidates first: select four rotated arena corners and a compact mouse-body seed. This is required before labelling."))
        prepare=QPushButton("Prepare ROI + automatic candidates"); prepare.clicked.connect(self._prepare); layout.addWidget(prepare)
        self.roi_edit.setPlaceholderText("Saved arena_roi.json (created automatically)")
        layout.addWidget(self._path_row(self.roi_edit, self._choose_roi))
        return group

    def _annotation_group(self) -> QGroupBox:
        group=QGroupBox("3. Expert labelling") ; grid=QGridLayout(group)
        torso=QPushButton("A. Adjust automatic torso polygons") ; torso.clicked.connect(self._annotate_torso)
        head=QPushButton("B. Correct head / reflection anchors") ; head.clicked.connect(self._annotate_head)
        grid.addWidget(torso,0,0); grid.addWidget(head,0,1)
        self.torso_labels_edit.setPlaceholderText("manual_torso_constraints.csv")
        self.head_labels_edit.setPlaceholderText("manual_head_anchor_labels.csv")
        grid.addWidget(self.torso_labels_edit,1,0); grid.addWidget(self.head_labels_edit,1,1)
        return group

    def _training_group(self) -> QGroupBox:
        group=QGroupBox("4. Train calibrated head model") ; layout=QVBoxLayout(group)
        layout.addWidget(QLabel("Trains a body-relative, time-block-validated model from anatomical head labels. Human labels remain hard overrides."))
        button=QPushButton("Train / update calibration model"); button.clicked.connect(self._train); layout.addWidget(button)
        layout.addWidget(self._path_row(self.model_edit,self._choose_model)); return group

    def _inference_group(self) -> QGroupBox:
        group=QGroupBox("5. Analyse and visualise") ; layout=QVBoxLayout(group)
        self.recovery=QCheckBox("Pause only after sustained tracking loss for user recovery")
        run=QPushButton("Run full inference + annotated video"); run.setObjectName("primary"); run.clicked.connect(self._infer)
        layout.addWidget(self.recovery); layout.addWidget(run); return group

    def _style(self) -> None:
        self.setStyleSheet("""
        QMainWindow { background:#f4f6f8; color:#15202b; } QGroupBox { border:1px solid #b9c5d0; border-radius:5px; margin-top:12px; padding:10px; background:#ffffff; font-weight:600; } QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; color:#173f5f; } QLabel#title { font-size:25px; font-weight:700; color:#123b5d; } QLabel#subtitle { color:#52616b; } QLabel#status { font-weight:600; color:#123b5d; } QPushButton { padding:7px 10px; border:1px solid #9aaab8; border-radius:4px; background:#eef3f7; } QPushButton:hover { background:#dceaf2; } QPushButton#primary { background:#155a7a; color:white; font-weight:700; border:none; } QLineEdit { padding:5px; border:1px solid #b9c5d0; border-radius:3px; } QPlainTextEdit { background:#111a22; color:#d8e5ef; font-family:Consolas; font-size:11px; } QProgressBar { height:8px; border:0; background:#dce3e8; } QProgressBar::chunk { background:#247ba0; }""")

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select mouse video", str(PROJECT_ROOT / "data"), "Video files (*.avi *.mp4 *.mov *.mkv)")
        if path:
            self.video_edit.setText(path); self.output_edit.setText(str(ROOT / "results" / f"{Path(path).stem}_gui"))

    def _choose_output(self) -> None:
        path=QFileDialog.getExistingDirectory(self,"Select output directory",str(ROOT / "results"));
        if path:self.output_edit.setText(path)
    def _choose_roi(self) -> None:
        path,_=QFileDialog.getOpenFileName(self,"Select ROI JSON",str(ROOT / "results"),"JSON (*.json)");
        if path:self.roi_edit.setText(path)
    def _choose_model(self) -> None:
        path,_=QFileDialog.getOpenFileName(self,"Select trained model",str(ROOT / "results"),"Model (*.joblib)");
        if path:self.model_edit.setText(path)

    def _values(self) -> dict[str, str]:
        video=self.video_edit.text().strip(); output=self.output_edit.text().strip()
        if not video or not Path(video).is_file(): raise ValueError("Choose an existing video file.")
        if not output: raise ValueError("Choose an output directory.")
        return {"video":video,"output":output,"roi":self.roi_edit.text().strip(),"model":self.model_edit.text().strip(),"head_labels":self.head_labels_edit.text().strip(),"torso_labels":self.torso_labels_edit.text().strip(),"width":str(self.width_spin.value()),"height":str(self.height_spin.value())}

    def _launch(self, script: str, args: list[str], label: str) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self,"Task active","A task is already running. Wait for completion before starting another."); return
        self.process=QProcess(self); self.process.setWorkingDirectory(str(ROOT)); self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output); self.process.finished.connect(self._finished)
        self.status.setText(label); self.progress.setRange(0,0); self.log.appendPlainText(f"\n$ {sys.executable} {script} {' '.join(args)}")
        self.process.start(sys.executable,[str(CODE/script),*args])
        if not self.process.waitForStarted(3000): self._fail("Could not start worker process")

    def _read_output(self) -> None:
        if self.process:self.pending_log.append(bytes(self.process.readAllStandardOutput()).decode(errors="replace").rstrip())
    def _flush_log(self) -> None:
        if self.pending_log:self.log.appendPlainText("\n".join(self.pending_log)); self.pending_log.clear()
    def _finished(self, code: int, status) -> None:
        self._flush_log(); self.progress.setRange(0,1); self.progress.setValue(1)
        self.status.setText("Completed successfully." if code==0 else f"Failed (exit code {code}). See process log.")
        self.process=None
    def _fail(self,text:str)->None: self.status.setText(text); self.progress.setRange(0,1); self.process=None

    def _annotate_torso(self) -> None:
        try:
            v=self._values(); roi=v['roi'] or str(Path(v['output'])/'arena_roi.json'); comp=str(Path(v['output'])/'head_method_comparison.csv'); out=str(Path(v['output'])/'manual_torso_constraints.csv')
            self.torso_labels_edit.setText(out); self._launch('annotate_torso_constraints.py',["--input",v['video'],"--comparison-csv",comp,"--roi-json",roi,"--output",out,"--arena-width-cm",v['width'],"--arena-height-cm",v['height'],"--max-labels","50"],"Annotate torso polygons (external OpenCV window)…")
        except ValueError as e: QMessageBox.warning(self,"Missing input",str(e))
    def _prepare(self) -> None:
        try:
            v=self._values(); args=["--input",v['video'],"--output-dir",v['output'],"--arena-width-cm",v['width'],"--arena-height-cm",v['height'],"--prepare-only"]
            if v['roi']: args += ["--roi-json",v['roi']]
            self.roi_edit.setText(str(Path(v['output'])/'arena_roi.json'))
            self._launch('process_new_video.py',args,"Preparing ROI and automatic candidates…")
        except ValueError as e: QMessageBox.warning(self,"Missing input",str(e))
    def _annotate_head(self) -> None:
        try:
            v=self._values(); roi=v['roi'] or str(Path(v['output'])/'arena_roi.json'); comp=str(Path(v['output'])/'head_method_comparison.csv'); out=str(Path(v['output'])/'manual_head_anchor_labels.csv')
            self.head_labels_edit.setText(out); self._launch('annotate_head_anchors.py',["--input",v['video'],"--comparison-csv",comp,"--roi-json",roi,"--output",out,"--arena-width-cm",v['width'],"--arena-height-cm",v['height'],"--max-labels","100"],"Annotate anatomical head / reflection (external OpenCV window)…")
        except ValueError as e: QMessageBox.warning(self,"Missing input",str(e))
    def _train(self) -> None:
        try:
            v=self._values(); labels=v['head_labels'] or str(Path(v['output'])/'manual_head_anchor_labels.csv'); model=str(Path(v['output'])/'head_calibrator.joblib'); self.model_edit.setText(model)
            self._launch('train_head_calibrator.py',["--comparison-csv",str(Path(v['output'])/'head_method_comparison.csv'),"--labels",labels,"--model-output",model,"--metrics-output",str(Path(v['output'])/'head_calibrator_metrics.json')],"Training calibration model…")
        except ValueError as e: QMessageBox.warning(self,"Missing input",str(e))
    def _infer(self) -> None:
        try:
            v=self._values()
            if not v['model'] or not Path(v['model']).is_file(): raise ValueError("Select a trained .joblib model before inference.")
            args=["--input",v['video'],"--model",v['model'],"--output-dir",v['output'],"--arena-width-cm",v['width'],"--arena-height-cm",v['height']]
            if v['roi']: args += ["--roi-json",v['roi']]
            if self.recovery.isChecked(): args += ["--interactive-recovery"]
            self._launch('process_new_video.py',args,"Running inference and annotated-video export…")
        except ValueError as e: QMessageBox.warning(self,"Missing input",str(e))
    def _open_output(self) -> None:
        path=self.output_edit.text().strip()
        if path: QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    def _open_result_video(self) -> None:
        path=Path(self.output_edit.text().strip())/'annotated_inference.mp4'
        if path.is_file(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else: QMessageBox.information(self,"No result yet","Run inference first; annotated_inference.mp4 will then be available.")
    def _save_project(self) -> None:
        try:
            v=self._values(); state=ProjectState(v['video'],v['output'],v['roi'],v['head_labels'],v['torso_labels'],v['model'],float(v['width']),float(v['height']))
            destination=Path(v['output'])/'gui_project.json'; Path(v['output']).mkdir(parents=True,exist_ok=True); destination.write_text(json.dumps(asdict(state),indent=2),encoding='utf-8'); self.status.setText(f"Saved project state: {destination.name}")
        except ValueError as e: QMessageBox.warning(self,"Missing input",str(e))


def main() -> None:
    app=QApplication(sys.argv); app.setApplicationName("Behavior Analyze")
    window=WorkflowWindow(); window.show(); sys.exit(app.exec())

if __name__ == "__main__": main()
