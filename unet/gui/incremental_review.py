#!/usr/bin/env python3
"""Review a finished video's failure candidates before incremental training."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

from core.label_compat import video_mask

UNET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNET_ROOT.parent
TORSO_EDITOR = REPO_ROOT / "traditional" / "code" / "annotate_torso_constraints.py"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--trajectory", required=True)
    p.add_argument("--mask-video", required=True)
    p.add_argument("--overlay-video", default="")
    p.add_argument("--roi-json", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--heads", required=True)
    p.add_argument("--candidate-csv", required=True)
    p.add_argument("--session-json", required=True)
    p.add_argument("--arena-width-cm", type=float, default=25.0)
    p.add_argument("--arena-height-cm", type=float, default=30.0)
    p.add_argument("--rotate", type=int, default=0)
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--initial-small-ratio", type=float, default=.65)
    return p


def opposite_head_frames(data: pd.DataFrame) -> list[tuple[int, float]]:
    required = ["body_x_cm", "body_y_cm", "head_x_cm", "head_y_cm",
                "reflection_x_cm", "reflection_y_cm"]
    if any(column not in data for column in required):
        return []
    values = data[required].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    body, head, reflection = values[:, :2], values[:, 2:4], values[:, 4:6]
    hv, rv = head - body, reflection - body
    hn = np.linalg.norm(hv, axis=1); rn = np.linalg.norm(rv, axis=1)
    cosine = np.sum(hv * rv, axis=1) / np.maximum(hn * rn, 1e-6)
    valid = np.isfinite(values).all(axis=1) & (hn >= .35) & (rn >= .25) & (cosine < -.20)
    frames = pd.to_numeric(data.get("frame"), errors="coerce").to_numpy()
    result = [(int(frames[i]), float(cosine[i])) for i in np.flatnonzero(valid)
              if np.isfinite(frames[i])]
    result.sort(key=lambda item: item[1])
    return result


def spaced_worst(rows: list[tuple[int, float]], maximum: int,
                 minimum_gap: int = 4) -> list[tuple[int, float]]:
    chosen: list[tuple[int, float]] = []
    for row in sorted(rows, key=lambda item: item[1]):
        if all(abs(row[0] - old[0]) >= minimum_gap for old in chosen):
            chosen.append(row)
            if len(chosen) >= maximum:
                break
    return sorted(chosen)


class Review:
    COLS, ROWS = 4, 3
    PAGE_SIZE = COLS * ROWS

    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root, self.args = root, args
        self.video = Path(args.video)
        self.trajectory = pd.read_csv(args.trajectory).drop_duplicates("frame", keep="last")
        self.mask_areas: np.ndarray | None = None
        self.normal_area = 1.0
        self.small_rows: list[tuple[int, float]] = []
        self.head_rows = spaced_worst(
            opposite_head_frames(self.trajectory), args.max_candidates)
        self.mode = tk.StringVar(value="small_mask")
        self.ratio = tk.DoubleVar(value=float(args.initial_small_ratio))
        self.status = tk.StringVar(value="正在后台扫描 Mask 面积…")
        self.progress_text = tk.StringVar(value="")
        self.page = 0
        self.photos: list[ImageTk.PhotoImage] = []
        self.pending_refresh: str | None = None
        self.editor_running = False
        self.exit_code = 1
        self.initial_torso = self.corrected_frames("small_mask")
        self.initial_heads = self.corrected_frames("head_opposite")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        threading.Thread(target=self._scan_masks, daemon=True).start()
        self.root.after(100, self._poll_scan)

    def _build(self) -> None:
        self.root.title("新视频增量训练复查")
        self.root.geometry("1120x790"); self.root.minsize(900, 650)
        bar = ttk.Frame(self.root); bar.pack(fill="x", padx=10, pady=8)
        ttk.Label(bar, text="候选类型").pack(side="left")
        modes = ttk.Combobox(
            bar, textvariable=self.mode, state="readonly", width=25,
            values=["small_mask", "head_opposite"])
        modes.pack(side="left", padx=5); modes.bind("<<ComboboxSelected>>", self._mode_changed)
        ttk.Label(bar, text="小 Mask 阈值：正常面积的").pack(side="left", padx=(15, 2))
        self.scale = ttk.Scale(
            bar, from_=.20, to=1.00, variable=self.ratio,
            command=lambda _value: self._schedule_threshold())
        self.scale.pack(side="left", fill="x", expand=True, padx=4)
        self.ratio_label = ttk.Label(bar, width=6)
        self.ratio_label.pack(side="left")
        self.edit_button = ttk.Button(bar, text="开始逐张修改", command=self.edit_current)
        self.edit_button.pack(side="right", padx=5)

        info = ttk.Frame(self.root); info.pack(fill="x", padx=12)
        ttk.Label(info, textvariable=self.status, foreground="#185a8d").pack(side="left")
        ttk.Label(info, textvariable=self.progress_text, foreground="#176b3a").pack(side="right")

        self.grid = ttk.Frame(self.root); self.grid.pack(fill="both", expand=True, padx=10, pady=8)
        nav = ttk.Frame(self.root); nav.pack(fill="x", padx=10, pady=(0, 9))
        ttk.Button(nav, text="取消，不训练", command=self.cancel).pack(side="left")
        ttk.Button(nav, text="上一页", command=lambda: self.move(-1)).pack(side="left", padx=8)
        ttk.Button(nav, text="下一页", command=lambda: self.move(1)).pack(side="left")
        ttk.Button(nav, text="完成复查并开始增量训练",
                   command=self.finish).pack(side="right")
        ttk.Label(nav, text="只有拖动/确认过的帧会写入训练标签",
                  foreground="#8a4a00").pack(side="right", padx=12)
        self._refresh_ratio_label()

    def _scan_masks(self) -> None:
        cap = cv2.VideoCapture(str(self.args.mask_video))
        areas = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            areas.append(int(np.count_nonzero(gray > 127)))
        cap.release()
        self.mask_areas = np.asarray(areas, dtype=float)

    def _poll_scan(self) -> None:
        if self.mask_areas is None:
            self.root.after(100, self._poll_scan); return
        positive = self.mask_areas[self.mask_areas > 0]
        if not len(positive):
            self.status.set("Mask 视频中没有有效前景")
            self.normal_area = 1.0
        else:
            low, high = np.percentile(positive, [20, 80])
            central = positive[(positive >= low) & (positive <= high)]
            self.normal_area = float(np.median(central if len(central) else positive))
            self.status.set(f"Mask 扫描完成；正常面积基准 {self.normal_area:.0f} px")
        self.recalculate()

    def _schedule_threshold(self) -> None:
        self._refresh_ratio_label()
        if self.pending_refresh:
            self.root.after_cancel(self.pending_refresh)
        self.pending_refresh = self.root.after(120, self.recalculate)

    def _refresh_ratio_label(self) -> None:
        self.ratio_label.configure(text=f"{100*self.ratio.get():.0f}%")

    def recalculate(self) -> None:
        self.pending_refresh = None
        if self.mask_areas is not None:
            ratio = self.mask_areas / max(self.normal_area, 1.0)
            rows = [(int(frame), float(value)) for frame, value in enumerate(ratio)
                    if 0 < value < self.ratio.get()]
            self.small_rows = spaced_worst(rows, self.args.max_candidates)
        self.page = 0
        self._write_candidates()
        self.render()

    def _write_candidates(self) -> None:
        rows = [
            {"video": self.video.name, "frame": frame,
             "candidate_type": "small_mask", "score": score, "exclude": False}
            for frame, score in self.small_rows
        ] + [
            {"video": self.video.name, "frame": frame,
             "candidate_type": "head_opposite", "score": score, "exclude": False}
            for frame, score in self.head_rows
        ]
        path = Path(self.args.candidate_csv); path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["video", "frame", "candidate_type", "score", "exclude"]).to_csv(
            path, index=False)

    def rows(self) -> list[tuple[int, float]]:
        return self.small_rows if self.mode.get() == "small_mask" else self.head_rows

    def _mode_changed(self, _event=None) -> None:
        self.page = 0; self.render()

    def move(self, delta: int) -> None:
        pages = max(1, int(np.ceil(len(self.rows()) / self.PAGE_SIZE)))
        self.page = min(max(0, self.page + delta), pages - 1)
        self.render()

    def corrected_frames(self, label_type: str) -> set[int]:
        path = Path(self.args.labels if label_type == "small_mask" else self.args.heads)
        if not path.is_file():
            return set()
        try:
            data = pd.read_csv(path)
            if "video" in data:
                data = data.loc[video_mask(data.video, self.video)]
            source = data.get("source", pd.Series("", index=data.index)).astype(str)
            wanted = ("incremental_small_mask" if label_type == "small_mask"
                      else "incremental_head_reflection")
            return set(pd.to_numeric(data.loc[source.str.contains(wanted, na=False), "frame"],
                                     errors="coerce").dropna().astype(int))
        except (OSError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return set()

    def render(self) -> None:
        for widget in self.grid.winfo_children():
            widget.destroy()
        self.photos.clear()
        rows = self.rows()
        corrected = self.corrected_frames(self.mode.get())
        start = self.page * self.PAGE_SIZE
        subset = rows[start:start + self.PAGE_SIZE]
        pages = max(1, int(np.ceil(len(rows) / self.PAGE_SIZE)))
        self.progress_text.set(
            f"候选 {len(rows)}｜已修改 {len(corrected & {f for f, _ in rows})}｜"
            f"第 {self.page+1}/{pages} 页")
        if not subset:
            ttk.Label(self.grid, text="当前阈值下没有候选帧",
                      font=("Microsoft YaHei UI", 13)).grid(row=0, column=0, padx=30, pady=80)
            return
        source_path = (Path(self.args.overlay_video)
                       if self.mode.get() == "head_opposite" and
                       Path(self.args.overlay_video).is_file() else self.video)
        cap = cv2.VideoCapture(str(source_path))
        mask_cap = cv2.VideoCapture(str(self.args.mask_video))
        for index, (frame, score) in enumerate(subset):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
            if not ok:
                continue
            if self.mode.get() == "small_mask":
                mask_cap.set(cv2.CAP_PROP_POS_FRAMES, frame); mok, mask = mask_cap.read()
                if mok:
                    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    green = image.copy(); green[gray > 127] = (0, 220, 0)
                    image = cv2.addWeighted(image, .68, green, .32, 0)
            image = cv2.cvtColor(cv2.resize(image, (260, 175)), cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(image)); self.photos.append(photo)
            box = ttk.Frame(self.grid, relief="solid", borderwidth=1)
            box.grid(row=index // self.COLS, column=index % self.COLS,
                     padx=5, pady=5, sticky="nsew")
            ttk.Label(box, image=photo).pack()
            mark = " ✓已修改" if frame in corrected else ""
            metric = (f"面积={100*score:.1f}%" if self.mode.get() == "small_mask"
                      else f"方向 cos={score:.2f}")
            ttk.Label(box, text=f"frame {frame}  {metric}{mark}",
                      foreground="#176b3a" if mark else "#333").pack(pady=2)
        cap.release(); mask_cap.release()
        for column in range(self.COLS):
            self.grid.columnconfigure(column, weight=1)

    def edit_current(self) -> None:
        if self.editor_running or not self.rows():
            return
        self._write_candidates()
        if self.mode.get() == "small_mask":
            command = [
                sys.executable, str(TORSO_EDITOR), "--input", str(self.video),
                "--comparison-csv", self.args.trajectory,
                "--roi-json", self.args.roi_json, "--output", self.args.labels,
                "--arena-width-cm", str(self.args.arena_width_cm),
                "--arena-height-cm", str(self.args.arena_height_cm),
                "--max-labels", str(self.args.max_candidates),
                "--mask-video", self.args.mask_video,
                "--candidate-csv", self.args.candidate_csv, "--only-modified"]
        else:
            command = [
                sys.executable, str(UNET_ROOT / "annotate_head_results.py"),
                "--video", str(self.video), "--trajectory", self.args.trajectory,
                "--roi-json", self.args.roi_json, "--heads", self.args.heads,
                "--mask-video", self.args.mask_video, "--torso-labels", self.args.labels,
                "--arena-width-cm", str(self.args.arena_width_cm),
                "--arena-height-cm", str(self.args.arena_height_cm),
                "--max-labels", str(self.args.max_candidates),
                "--rotate", str(self.args.rotate),
                "--candidate-csv", self.args.candidate_csv, "--only-modified"]
        self.editor_running = True; self.edit_button.configure(state="disabled")
        self.status.set("编辑器运行中：S 保存并切换下一张；关闭后拼接页会刷新")

        def worker():
            code = subprocess.run(command, cwd=str(REPO_ROOT)).returncode
            self.root.after(0, lambda: self._editor_done(code))
        threading.Thread(target=worker, daemon=True).start()

    def _editor_done(self, code: int) -> None:
        self.editor_running = False; self.edit_button.configure(state="normal")
        self.status.set("编辑结果已实时写入 CSV" if code == 0 else f"编辑器退出码 {code}")
        self.render()

    def finish(self) -> None:
        torso = self.corrected_frames("small_mask") - self.initial_torso
        heads = self.corrected_frames("head_opposite") - self.initial_heads
        if not torso and not heads:
            messagebox.showwarning("没有新增修改",
                                   "没有任何帧被实际修改或确认，因此不会启动训练。")
            return
        session = {
            "video": str(self.video.resolve()), "small_mask_ratio": self.ratio.get(),
            "torso_frames": sorted(torso), "head_frames": sorted(heads),
            "candidate_csv": str(Path(self.args.candidate_csv).resolve())}
        path = Path(self.args.session_json); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        self.exit_code = 0; self.root.destroy()

    def cancel(self) -> None:
        if self.editor_running:
            messagebox.showinfo("编辑器仍在运行", "请先关闭当前 OpenCV 编辑器。")
            return
        self.exit_code = 1; self.root.destroy()


def main() -> None:
    args = parser().parse_args()
    for required in (args.video, args.trajectory, args.mask_video, args.roi_json):
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    root = tk.Tk(); app = Review(root, args); root.mainloop()
    raise SystemExit(app.exit_code)


if __name__ == "__main__":
    main()
