#!/usr/bin/env python3
"""GUI wizard for the whole run_unet.py workflow.

Two modes (notebook tabs):
  Training:   roi -> compare -> screen -> annotate -> prepare -> train
  Processing: infer -> head
Every step launches exactly the same command the CLI launcher would run, in
a background thread, streaming output to the log pane. Step state is read
from the artifacts on disk (ROI JSON, comparison.csv, screening.csv,
labels, dataset, model, trajectory CSVs); a video picker plus an
"apply to all videos" toggle control whether loop steps run on one or on
every recording. New videos can be added from the UI and get their ROI
through the corner picker automatically.

Run: python unet/run_unet.py ui
"""
from __future__ import annotations

from collections import deque
import subprocess
import sys
import threading
from argparse import Namespace
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import run_unet as R

FONT = ("Microsoft YaHei UI", 10)
DONE, READY, WARN = "done", "ready", "warn"
LOOP_STEPS = ("roi", "compare", "screen", "annotate", "prepare")
# Log throttling: keep the UI responsive during long runs (10-min videos,
# training). The widget keeps at most LOG_MAX_LINES, each pump inserts at
# most LOG_PUMP_BATCH lines, and the worker-side queue is bounded so a
# chatty child process cannot grow memory without limit.
LOG_MAX_LINES = 4000
LOG_PUMP_BATCH = 200
LOG_QUEUE_MAX = 4000

TRAIN_STEPS = [
    ("roi",      "① ROI 四角标注", "新视频:点竞技场 4 个角"),
    ("compare",  "② 生成对比数据", "生成 head_method_comparison.csv"),
    ("screen",   "③ 帧筛选",      "排除无鼠 / 实验者干预帧"),
    ("annotate", "④ 多边形标注",   "修正身体轮廓(需 ②③)"),
    ("prepare",  "⑤ 数据集导出",   "生成 images/masks(需 ④)"),
    ("train",    "⑥ 模型训练",     "训练 U-Net(需 ⑤)"),
]
PROC_STEPS = [
    ("infer", "① 分割推理",      "全帧 UNet 分割 -> mask/overlay"),
    ("head",  "② 头+身体跟踪",   "mask 质心 + 反光点头部(需 ①)"),
]
DEPENDENCIES = {
    "annotate": ("compare", "screen"),
    "prepare":  ("annotate",),
    "train":    ("prepare",),
    "head":     ("infer",),
}
STATE_TEXT = {DONE: "✓ 完成", READY: "○ 待运行", WARN: "△ 依赖缺失"}
STATE_COLOR = {DONE: "#1a7f37", READY: "#0b57d0", WARN: "#b06000"}


def build_args() -> Namespace:
    return Namespace(
        screening=R.SCREENING, labels=R.LABELS, dataset=R.DATASET,
        model=R.MODEL, infer_out=R.INFER_OUT,
        arena_width_cm=R.ARENA_WIDTH_CM, arena_height_cm=R.ARENA_HEIGHT_CM,
        max_labels=100, per_video=40, junk=20, size=256,
        epochs=80, batch_size=8, threshold=0.5,
    )


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.args = build_args()
        self.videos = [dict(c) for c in R.VIDEOS]
        self.current = 0
        self.proc: subprocess.Popen | None = None
        self.log_q: deque[str] = deque(maxlen=LOG_QUEUE_MAX)  # drops oldest
        self.log_lines = 0  # current line count of the log Text widget
        self.all_var = tk.BooleanVar(value=True)
        self.video_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        self.step_rows: dict[str, tuple[tk.Label, tk.Label]] = {}
        self._build_ui()
        self._fill_video_combo()
        self._start_log_pump()
        self.refresh()

    # ------------------------------------------------------------ paths ---
    def current_cfg(self) -> dict:
        return self.videos[self.current]

    def roi_path(self, cfg: dict) -> Path:
        p = Path(cfg.get("roi", ""))
        if p.is_file():
            return p
        return R.ROOT / "traditional" / "basic_rois" / f"{Path(cfg['video']).stem}_roi.json"

    def local_cfg(self, cfg: dict) -> dict:
        c = dict(cfg)
        c["roi"] = str(self.roi_path(cfg))
        # keep Path: annotate_cmd does cfg["compare_out"] / "file.csv"
        c["compare_out"] = Path(cfg.get("compare_out") or
                                R.ROOT / "traditional" / "results" / "basic_recognition" / Path(cfg["video"]).stem)
        return c

    def make_v(self) -> dict:
        return {"video": self.current_cfg()["video"], "model": self.args.model,
                "infer_out": self.args.infer_out, "dataset": self.args.dataset,
                "labels": self.args.labels, "screening": self.args.screening}

    # ----------------------------------------------------------- state ---
    def status_of(self, key: str) -> str:
        cfg = self.current_cfg()
        if key == "roi":
            ok = self.roi_path(cfg).is_file()
        elif key == "compare":
            out = cfg.get("compare_out") or R.ROOT / "traditional" / "results" / "basic_recognition" / Path(cfg["video"]).stem
            ok = (Path(out) / "head_method_comparison.csv").is_file()
        elif key == "screen":
            ok = self.args.screening.is_file()
        elif key == "annotate":
            ok = self.args.labels.is_file()
        elif key == "prepare":
            images = self.args.dataset / "images"
            ok = images.is_dir() and any(images.glob("*.png"))
        elif key == "train":
            ok = self.args.model.is_file()
        elif key == "infer":
            ok = (self.args.infer_out / "unet_trajectory.csv").is_file()
        elif key == "head":
            ok = (self.args.infer_out / "head_track_trajectory.csv").is_file()
        else:
            ok = False
        return DONE if ok else READY

    def refresh(self) -> None:
        # cache status_of per refresh: dependencies re-query the same keys
        cache: dict[str, str] = {}

        def st(key: str) -> str:
            if key not in cache:
                cache[key] = self.status_of(key)
            return cache[key]

        for key, title, hint in TRAIN_STEPS + PROC_STEPS:
            state = st(key)
            if state == READY and any(st(d) != DONE for d in DEPENDENCIES.get(key, ())):
                state = WARN
            label, badge = self.step_rows[key]
            label.config(text=title)
            badge.config(text=STATE_TEXT[state], fg=STATE_COLOR[state])
        self.root.update_idletasks()

    # -------------------------------------------------------- commands ---
    def commands_for(self, key: str) -> list[tuple[Path, list[str]]]:
        w = f"{self.args.arena_width_cm:.2f}"
        h = f"{self.args.arena_height_cm:.2f}"
        loop = self.all_var.get() and key in LOOP_STEPS
        cfgs = [self.local_cfg(c) for c in self.videos] if loop else [self.local_cfg(self.current_cfg())]
        if key == "roi":
            return [R.roi_cmd(c) for c in cfgs]
        if key == "compare":
            return [R.compare_cmd(c, w, h) for c in cfgs]
        if key == "screen":
            return [R.screen_cmd(c, w, h, self.args) for c in cfgs]
        if key == "annotate":
            return [R.annotate_cmd(c, w, h, self.args) for c in cfgs]
        if key == "prepare":
            return [R.prepare_cmd(c, w, h, self.args) for c in cfgs]
        if key == "train":
            return [R.train_cmd(self.make_v(), self.args)]
        if key == "infer":
            return [R.infer_cmd(self.make_v(), self.args)]
        if key == "head":
            cfg = {"video": self.make_v()["video"], "roi": self.roi_path(self.current_cfg())}
            return [R.head_cmd(self.make_v(), cfg, self.args)]
        return []

    # ---------------------------------------------------------- running ---
    def run_scripts(self, cmds: list[tuple[Path, list[str]]], then=None) -> None:
        if not cmds:
            return
        if self.proc is not None and self.proc.poll() is None:
            self.log("任务运行中，新任务已忽略（先停止当前任务）。\n")
            return
        self.status_var.set(f"运行 {len(cmds)} 个任务 ...")

        def worker() -> None:
            for script, argv in cmds:
                if not self._exec(script, argv):
                    break
            if then is not None:
                try:
                    then()
                except Exception as e:  # noqa: BLE001
                    self.log(f"ERROR in follow-up: {e}\n")
            self.root.after(0, lambda: (self.refresh(), self.status_var.set("完成(请查看日志)")))

        threading.Thread(target=worker, daemon=True).start()

    def _exec(self, script: Path, argv: list[str]) -> bool:
        self.log(f"$ python {Path(script).name} {' '.join(argv)}\n")
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", str(script), *argv],  # -u: unbuffered prints
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=str(R.ROOT))
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log(line)
            rc = self.proc.wait()
            self.proc = None
            self.log(f"\n--- 退出码 {rc} ---\n")
            return rc == 0
        except Exception as e:  # noqa: BLE001
            self.log(f"ERROR: {e}\n")
            self.proc = None
            return False

    def log(self, text: str) -> None:
        self.log_q.append(text)  # deque.append is atomic under the GIL

    def _start_log_pump(self) -> None:
        def pump() -> None:
            inserted = 0
            while inserted < LOG_PUMP_BATCH and self.log_q:
                try:
                    line = self.log_q.popleft()
                except IndexError:  # another thread drained it meanwhile
                    break
                self.log_text.insert(tk.END, line)
                self.log_lines += line.count("\n")
                inserted += 1
            if inserted:
                self._trim_log()
                self.log_text.see(tk.END)
            self.root.after(120, pump)
        self.root.after(120, pump)

    def _trim_log(self) -> None:
        """Keep the widget bounded: drop the oldest lines past LOG_MAX_LINES."""
        excess = self.log_lines - LOG_MAX_LINES
        if excess > 0:
            self.log_text.delete("1.0", f"{excess + 1}.0")
            self.log_lines -= excess

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.log("\n--- 已请求停止 ---\n")

    def run_step(self, key: str) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("任务运行中", "当前任务尚未结束，请先停止或等待完成。")
            return
        missing = [d for d in DEPENDENCIES.get(key, ()) if self.status_of(d) != DONE]
        if missing and not messagebox.askyesno(
                "依赖缺失",
                f"推荐先完成:{'、'.join(missing)}\n仍要运行 {key} 吗?"):
            return
        if key == "head" and not self.roi_path(self.current_cfg()).is_file():
            if not messagebox.askyesno("缺少 ROI", "当前视频没有 ROI JSON,先打开四角标注窗口?"):
                return
            self.status_var.set("ROI 标注 ...")
            self.run_scripts([R.roi_cmd(self.current_cfg())],
                             then=lambda: self._head_after_roi())
            return
        cmds = self.commands_for(key)
        self.run_scripts(cmds)

    def _head_after_roi(self) -> None:
        if not self.roi_path(self.current_cfg()).is_file():
            self.log("ROI 标注未保存,取消 head 任务。\n")
            return
        self.run_scripts(self.commands_for("head"))

    def add_video(self) -> None:
        f = filedialog.askopenfilename(
            title="选择新视频", initialdir=str(R.ROOT / "data"),
            filetypes=[("视频", "*.avi *.mp4 *.mov *.mkv")])
        if not f:
            return
        cfg = {"video": Path(f), "roi": "", "compare_out": ""}
        self.videos.append(cfg)
        self.current = len(self.videos) - 1
        self._fill_video_combo()
        self.refresh()
        if not self.roi_path(cfg).is_file():
            self.status_var.set("新视频缺少 ROI,自动打开四角标注 ...")
            self.run_scripts([R.roi_cmd(cfg)])

    # -------------------------------------------------------------- ui ---
    def _fill_video_combo(self) -> None:
        names = [Path(c["video"]).name for c in self.videos]
        self.video_combo["values"] = names
        self.video_var.set(names[self.current])

    def _on_video_change(self, _event=None) -> None:
        try:
            self.current = int(self.video_combo.current())
        except ValueError:
            self.current = 0
        self.refresh()

    def _step_row(self, parent, key: str, title: str, hint: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        label = ttk.Label(row, text=title, font=FONT, width=18, anchor="w")
        label.pack(side="left")
        ttk.Label(row, text=hint, font=("Microsoft YaHei UI", 9),
                  foreground="#666666", width=38, anchor="w").pack(side="left")
        badge = tk.Label(row, text="○ 待运行", font=FONT, width=10,
                         anchor="center",
                         bg=ttk.Style().lookup("TFrame", "background"))
        badge.pack(side="right", padx=(4, 0))
        ttk.Button(row, text="运行", width=8,
                   command=lambda k=key: self.run_step(k)).pack(side="right", padx=4)
        self.step_rows[key] = (label, badge)

    def _build_ui(self) -> None:
        root = self.root
        root.title("run_unet 流程向导")
        root.geometry("1040x760")
        root.minsize(900, 640)

        nb = ttk.Notebook(root)
        tab_train = ttk.Frame(nb)
        tab_proc = ttk.Frame(nb)
        nb.add(tab_train, text=" 训练模式 ")
        nb.add(tab_proc, text=" 处理模式 ")
        nb.pack(fill="both", expand=True, padx=8, pady=(6, 0))

        for spec in TRAIN_STEPS:
            self._step_row(tab_train, *spec)
        ttk.Separator(tab_train).pack(fill="x", pady=6)
        for spec in PROC_STEPS:
            self._step_row(tab_proc, *spec)

        # video bar (shared)
        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=8, pady=4)
        ttk.Label(bar, text="当前视频:", font=FONT).pack(side="left")
        self.video_combo = ttk.Combobox(bar, textvariable=self.video_var,
                                        state="readonly", width=52, font=FONT)
        self.video_combo.pack(side="left", padx=4)
        self.video_combo.bind("<<ComboboxSelected>>", self._on_video_change)
        ttk.Button(bar, text="添加新视频", command=self.add_video).pack(side="left", padx=4)
        ttk.Checkbutton(bar, text="循环步骤应用到全部视频", variable=self.all_var,
                        command=self.refresh).pack(side="left", padx=10)
        ttk.Button(bar, text="刷新状态", command=self.refresh).pack(side="right", padx=4)
        ttk.Button(bar, text="停止任务", command=self.stop).pack(side="right", padx=4)

        # status + log
        bottom = ttk.Frame(root)
        bottom.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        top = ttk.Frame(bottom)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.status_var, font=FONT,
                  foreground="#0b57d0").pack(side="left")
        ttk.Label(top, text="(依赖缺失的步骤点运行会先询问)", font=("Microsoft YaHei UI", 9),
                  foreground="#666666").pack(side="right")
        self.log_text = tk.Text(bottom, height=14, font=("Consolas", 9),
                                bg="#101418", fg="#d6e0e8", wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=(4, 0))


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()