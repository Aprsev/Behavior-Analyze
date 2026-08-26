#!/usr/bin/env python3
"""Production desktop GUI for fibre-aware U-Net training and analysis.

The UI is intentionally a thin controller around the same scripts used by
the CLI. Long video/GPU work always runs in a child process; Tk is touched
only by the main thread through a queue-driven state machine.
"""
from __future__ import annotations

from collections import deque
import json
import os
import queue
import subprocess
import sys
import threading
from argparse import Namespace
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import run_unet as R
from core.label_compat import as_bool, video_matches

FONT = ("Microsoft YaHei UI", 10)
SMALL = ("Microsoft YaHei UI", 9)
UNET_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = UNET_ROOT / ".ui_state.json"
LEGACY_VIDEO_FILE = UNET_ROOT / ".ui_videos.json"
LOG_MAX_LINES = 5000


def default_args() -> Namespace:
    return Namespace(
        screening=R.SCREENING, labels=R.LABELS, heads=R.HEADS,
        dataset=R.DATASET, model=R.MODEL, infer_out=R.INFER_OUT,
        arena_width_cm=R.ARENA_WIDTH_CM, arena_height_cm=R.ARENA_HEIGHT_CM,
        max_labels=10, per_video=10, junk=20, size=256,
        epochs=80, batch_size=8, lr=2e-3, calib_frames=20,
        threshold=.5, fibre_opening=5, reacquire_sec=.35, rotate=0,
    )


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.args = default_args()
        self.saved = self._read_state()
        self.videos = self._load_videos()
        self.current = min(int(self.saved.get("current", 0)), max(len(self.videos) - 1, 0))
        self.proc: subprocess.Popen | None = None
        self.running = False
        self.current_script: Path | None = None
        self.events: queue.Queue = queue.Queue()
        self.command_queue: deque[tuple[Path, list[str]]] = deque()
        self.cancelled = False
        self.task_success_callback = None
        self.action_buttons: list[ttk.Button] = []
        self.train_curve: dict[int, tuple[float, float]] = {}
        self.log_lines = 0

        self.video_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        self.model_var = tk.StringVar(value=str(self.saved.get("model", self.args.model)))
        self.output_var = tk.StringVar()
        self.dataset_var = tk.StringVar(value=str(self.saved.get("dataset", self.args.dataset)))
        self.threshold_var = tk.StringVar(value=str(self.saved.get("threshold", .5)))
        self.fibre_var = tk.StringVar(value=str(self.saved.get("fibre_opening", 5)))
        self.reacquire_var = tk.StringVar(value=str(self.saved.get("reacquire_sec", .35)))
        self.rotate_var = tk.IntVar(value=int(self.saved.get("rotate", 0)))
        self.epochs_var = tk.StringVar(value=str(self.saved.get("epochs", 80)))
        self.batch_var = tk.StringVar(value=str(self.saved.get("batch_size", 8)))
        self.lr_var = tk.StringVar(value=str(self.saved.get("lr", .002)))
        self.per_video_var = tk.StringVar(value=str(self.saved.get("per_video", 10)))
        self.all_train_var = tk.BooleanVar(value=bool(self.saved.get("all_train", True)))
        self.analysis_summary_var = tk.StringVar()
        self.training_summary_var = tk.StringVar()
        self.roi_summary_var = tk.StringVar()

        self._build_ui()
        self._fill_video_combo()
        self._select_current()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll_events)
        self.refresh()

    # ---------------------------------------------------------- state/files
    def _read_state(self) -> dict:
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.is_file() else {}
        except (OSError, json.JSONDecodeError, TypeError):
            return {}

    def _load_videos(self) -> list[dict]:
        rows = self.saved.get("videos") or []
        if not rows and LEGACY_VIDEO_FILE.is_file():
            try:
                rows = json.loads(LEGACY_VIDEO_FILE.read_text(encoding="utf-8")).get("videos") or []
            except (OSError, json.JSONDecodeError):
                rows = []
        out = []
        for row in rows:
            try:
                video = Path(row["video"])
            except (KeyError, TypeError):
                continue
            if video.is_file():
                out.append({"video": video, "roi": Path(row.get("roi") or ""),
                            "compare_out": Path(row.get("compare_out") or ""),
                            "output": Path(row.get("output") or "")})
        if not out:
            out = [dict(row) for row in R.VIDEOS if Path(row["video"]).is_file()]
        if not out:
            out = [{"video": R.ROOT / "data" / "select_video.avi",
                    "roi": Path(""), "compare_out": Path(""), "output": Path("")}]
        return out

    def _save_state(self) -> None:
        self._sync_args(show_error=False)
        data = {
            "current": self.current, "model": str(self.args.model),
            "dataset": str(self.args.dataset), "threshold": self.args.threshold,
            "fibre_opening": self.args.fibre_opening,
            "reacquire_sec": self.args.reacquire_sec, "rotate": self.args.rotate,
            "epochs": self.args.epochs, "batch_size": self.args.batch_size,
            "lr": self.args.lr, "per_video": self.args.per_video,
            "all_train": bool(self.all_train_var.get()),
            "videos": [{"video": str(v["video"]), "roi": str(v.get("roi") or ""),
                        "compare_out": str(v.get("compare_out") or ""),
                        "output": str(v.get("output") or "")} for v in self.videos],
        }
        try:
            STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self.log(f"WARNING: 无法保存 UI 设置: {exc}\n")

    def cfg(self) -> dict:
        return self.videos[self.current]

    def roi_path(self, cfg: dict | None = None) -> Path:
        cfg = cfg or self.cfg()
        saved = Path(cfg.get("roi") or "")
        if saved.is_file():
            return saved
        return R.ROOT / "traditional" / "basic_rois" / f"{Path(cfg['video']).stem}_roi.json"

    def compare_path(self, cfg: dict | None = None) -> Path:
        cfg = cfg or self.cfg()
        return Path(cfg.get("compare_out") or
                    R.ROOT / "traditional" / "results" / "basic_recognition" / Path(cfg["video"]).stem)

    def output_path(self, cfg: dict | None = None) -> Path:
        cfg = cfg or self.cfg()
        saved = Path(cfg.get("output") or "")
        return saved if str(saved) not in ("", ".") else R.ROOT / "results" / f"{Path(cfg['video']).stem}_unet"

    def local_cfg(self, cfg: dict) -> dict:
        return {"video": Path(cfg["video"]), "roi": self.roi_path(cfg),
                "compare_out": self.compare_path(cfg)}

    def make_v(self) -> dict:
        return {"video": Path(self.cfg()["video"]), "model": self.args.model,
                "infer_out": self.args.infer_out, "dataset": self.args.dataset,
                "labels": self.args.labels, "screening": self.args.screening,
                "heads": self.args.heads}

    def _sync_args(self, show_error: bool = True) -> bool:
        try:
            self.args.model = Path(self.model_var.get().strip())
            self.args.dataset = Path(self.dataset_var.get().strip())
            self.args.infer_out = Path(self.output_var.get().strip())
            self.args.threshold = min(.99, max(.01, float(self.threshold_var.get())))
            opening = max(3, int(self.fibre_var.get()))
            self.args.fibre_opening = opening if opening % 2 else opening + 1
            self.args.reacquire_sec = min(10., max(.05, float(self.reacquire_var.get())))
            self.args.rotate = int(self.rotate_var.get())
            self.args.epochs = max(1, int(self.epochs_var.get()))
            self.args.batch_size = max(1, int(self.batch_var.get()))
            requested_lr = float(self.lr_var.get())
            self.args.lr = min(3e-3, max(1e-6, requested_lr))
            if requested_lr != self.args.lr:
                self.lr_var.set(f"{self.args.lr:g}")
            self.args.per_video = max(1, int(self.per_video_var.get()))
            self.args.max_labels = self.args.per_video
            self.cfg()["output"] = self.args.infer_out
            return True
        except (ValueError, tk.TclError) as exc:
            if show_error:
                messagebox.showerror("参数错误", f"请检查阈值、光纤宽度、重捕获时间及训练参数。\n{exc}")
            return False

    # --------------------------------------------------------------- status
    @staticmethod
    def _same_path(left, right) -> bool:
        try:
            return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))
        except OSError:
            return False

    def _metadata_matches(self, filename: str) -> bool:
        path = self.output_path() / filename
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            return self._same_path(meta.get("input", ""), self.cfg()["video"])
        except (OSError, json.JSONDecodeError):
            return False

    def refresh(self) -> None:
        cfg = self.cfg(); video = Path(cfg["video"]); roi = self.roi_path(cfg)
        self.roi_summary_var.set(f"ROI: {roi}" if roi.is_file() else "ROI: 未设置")
        if self._metadata_matches("head_track_metadata.json"):
            try:
                meta = json.loads((self.output_path() / "head_track_metadata.json").read_text(encoding="utf-8"))
                self.analysis_summary_var.set(
                    f"当前视频已完成：{meta.get('frames', 0)} 帧，头部有效率 {meta.get('head_valid_percent', 0):.1f}%")
            except (OSError, json.JSONDecodeError):
                self.analysis_summary_var.set("当前视频结果存在")
        else:
            self.analysis_summary_var.set("当前视频尚未完成完整分析")
        history = self.args.model.parent / "training_history.json"
        if self.args.model.is_file():
            try:
                d = json.loads(history.read_text(encoding="utf-8"))
                promoted = "已晋升" if d.get("candidate_promoted", True) else "保留旧模型"
                self.training_summary_var.set(
                    f"模型可用｜Dice {100 * float(d['best_val_dice']):.1f}%｜"
                    f"Head {d.get('best_head_error_px', '?')} px｜"
                    f"Reflection {d.get('best_reflection_error_px', '?')} px｜{promoted}｜"
                    f"训练 {d.get('train_count', '?')} / 验证 {d.get('val_count', '?')}")
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                self.training_summary_var.set("模型文件存在")
        else:
            self.training_summary_var.set("模型不存在：请先训练或选择 checkpoint")
        self.root.title(f"Behavior Analyze · U-Net v3 — {video.name}")

    # -------------------------------------------------------------- commands
    def _training_cfgs(self) -> list[dict]:
        rows = self.videos if self.all_train_var.get() else [self.cfg()]
        # The project list may also contain inference-only recordings. Do not
        # let an unlabeled new video block or pollute dataset rebuilding.
        if self.args.labels.is_file():
            try:
                labels = pd.read_csv(self.args.labels)
                if "video" in labels.columns:
                    labelled_names = list(labels.loc[
                        labels.get("polygon_px", pd.Series(index=labels.index, dtype=object)).notna(),
                        "video"].astype(str))
                    rows = [row for row in rows
                            if any(video_matches(value, row["video"]) for value in labelled_names)]
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                pass
        return [self.local_cfg(row) for row in rows if Path(row["video"]).is_file()]

    def _require_analysis_inputs(self) -> bool:
        if not Path(self.cfg()["video"]).is_file():
            messagebox.showerror("视频不存在", str(self.cfg()["video"])); return False
        if not self.roi_path().is_file():
            messagebox.showerror("缺少 ROI", "请先点击“框选 ROI”或选择已有 ROI JSON。"); return False
        if not self.args.model.is_file():
            messagebox.showerror("缺少模型", "请选择可用的 U-Net checkpoint，或先在训练页训练。"); return False
        return True

    def analyze(self) -> None:
        if not self._sync_args() or not self._require_analysis_inputs():
            return
        self.args.infer_out.mkdir(parents=True, exist_ok=True)
        cmd = R.head_cmd(self.make_v(), {"video": self.cfg()["video"], "roi": self.roi_path()}, self.args)
        self.run_commands([cmd], "完整分析")

    def mask_preview(self) -> None:
        if not self._sync_args() or not self.args.model.is_file():
            messagebox.showerror("缺少模型", "请选择可用的 U-Net checkpoint。"); return
        self.args.infer_out.mkdir(parents=True, exist_ok=True)
        self.run_commands([R.infer_cmd(self.make_v(), self.args)], "仅分割预览")

    def incremental_train(self) -> None:
        """Review this finished video's failures, then fine-tune the current model."""
        if not self._sync_args() or not self._require_analysis_inputs():
            return
        cfg = self.local_cfg(self.cfg())
        output = self.output_path()
        required = [output / "head_track_trajectory.csv",
                    output / "mouse_miniscope_mask.mp4"]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            messagebox.showerror(
                "缺少分析结果",
                "请先运行“一键完整分析”。缺少：\n" + "\n".join(missing))
            return
        if not (self.args.dataset / "dataset.json").is_file():
            messagebox.showerror(
                "缺少原训练数据集",
                "增量训练需要保留原数据集，避免只用一个新视频导致灾难性遗忘。\n"
                "请先在“使用已有标注训练”页重建原训练数据集。")
            return
        if not messagebox.askyesno(
                "增量训练",
                "将先筛选小 Mask 和头/反光点反侧帧。\n"
                "只有实际修改或确认的帧会加入训练数据；随后从当前模型继续训练，"
                "不会覆盖原 checkpoint。继续吗？"):
            return
        w, h = f"{self.args.arena_width_cm:.2f}", f"{self.args.arena_height_cm:.2f}"
        self.train_curve.clear(); self.draw_curve()
        commands = [
            R.incremental_review_cmd(self.make_v(), cfg, self.args),
            R.prepare_cmd(cfg, w, h, self.args),
            R.train_cmd(self.make_v(), self.args)]
        self.run_commands(commands, "新视频复查并增量训练")

    def rebuild_dataset(self, train_after: bool = False) -> None:
        if not self._sync_args():
            return
        if not self.args.labels.is_file():
            messagebox.showerror("缺少标注", f"未找到已有标注：\n{self.args.labels}"); return
        cfgs = self._training_cfgs()
        if not cfgs:
            messagebox.showerror("没有训练视频", "视频列表中没有与标注 CSV 匹配的录像。"); return
        missing_roi = [Path(c["video"]).name for c in cfgs if not Path(c["roi"]).is_file()]
        if missing_roi:
            messagebox.showerror("缺少 ROI", "这些训练视频没有 ROI：\n" + "\n".join(missing_roi)); return
        w, h = f"{self.args.arena_width_cm:.2f}", f"{self.args.arena_height_cm:.2f}"
        commands = [R.prepare_cmd(c, w, h, self.args) for c in cfgs]
        if train_after:
            self.train_curve.clear(); self.draw_curve()
            commands.append(R.train_cmd(self.make_v(), self.args))
        self.run_commands(commands, "重建数据集并训练" if train_after else "重建数据集")

    def train(self) -> None:
        if not self._sync_args():
            return
        if not (self.args.dataset / "dataset.json").is_file():
            messagebox.showerror("缺少数据集", "请先点击“重建数据集”。"); return
        self.train_curve.clear(); self.draw_curve()
        self.run_commands([R.train_cmd(self.make_v(), self.args)], "模型训练")

    def advanced_step(self, key: str) -> None:
        if not self._sync_args():
            return
        cfg = self.local_cfg(self.cfg()); w = f"{self.args.arena_width_cm:.2f}"; h = f"{self.args.arena_height_cm:.2f}"
        if key != "roi" and not Path(cfg["roi"]).is_file():
            messagebox.showerror("缺少 ROI", "请先框选当前视频 ROI。"); return
        if key == "roi": commands = [R.roi_cmd(cfg)]
        elif key == "compare": commands = [R.compare_cmd(cfg, w, h)]
        elif key == "screen": commands = [R.screen_cmd(cfg, w, h, self.args)]
        elif key == "annotate":
            if not (cfg["compare_out"] / "head_method_comparison.csv").is_file():
                messagebox.showerror("缺少对比数据", "请先运行“生成传统对比数据”。"); return
            commands = [R.annotate_cmd(cfg, w, h, self.args)]
        elif key == "review":
            if not self.args.labels.is_file():
                messagebox.showerror("缺少标注", "尚未找到已有多边形标注 CSV。"); return
            if not (cfg["compare_out"] / "head_method_comparison.csv").is_file():
                messagebox.showerror("缺少对比数据", "请先运行“生成传统对比数据”。"); return
            # Immediately synchronize corrected/excluded rows into the
            # training dataset so stale PNG masks cannot survive a review.
            commands = [R.annotate_cmd(cfg, w, h, self.args, review_existing=True),
                        R.prepare_cmd(cfg, w, h, self.args)]
        elif key == "head_result":
            trajectory = self.output_path() / "head_track_trajectory.csv"
            if not trajectory.is_file():
                messagebox.showerror("缺少头部结果", "请先在“一键分析”页面运行完整分析。"); return
            commands = [R.annotate_head_results_cmd(self.make_v(), cfg, self.args),
                        R.prepare_cmd(cfg, w, h, self.args)]
        elif key == "calibrate":
            if not self.args.model.is_file():
                messagebox.showerror("缺少模型", "请先训练模型。"); return
            commands = [R.calibrate_cmd(self.make_v(), cfg, self.args)]
            commands += [R.prepare_cmd(c, w, h, self.args) for c in self._training_cfgs()]
            commands += [R.train_cmd(self.make_v(), self.args)]
        else: return
        self.run_commands(commands, key)

    def show_annotation_stats(self) -> None:
        lines = []
        try:
            labels = pd.read_csv(self.args.labels) if self.args.labels.is_file() else pd.DataFrame()
            heads = pd.read_csv(self.args.heads) if self.args.heads.is_file() else pd.DataFrame()
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            messagebox.showerror("标注文件错误", str(exc)); return
        for cfg in self.videos:
            name = Path(cfg["video"]).name
            torso_rows = labels.loc[labels.video.map(lambda value: video_matches(value, name))] if "video" in labels else labels
            head_rows = heads.loc[heads.video.map(lambda value: video_matches(value, name))] if "video" in heads else heads
            head_present = int(head_rows.get("head_present", pd.Series(True, index=head_rows.index)).map(
                lambda value: str(value).strip().casefold() in {"1", "true", "yes", "y"}).sum())
            reflection_present = int(head_rows.get("reflection_present", pd.Series(True, index=head_rows.index)).map(
                lambda value: str(value).strip().casefold() in {"1", "true", "yes", "y"}).sum())
            head_verified = int(head_rows.get("head_verified", pd.Series(False, index=head_rows.index)).map(as_bool).sum())
            reflection_verified = int(head_rows.get(
                "reflection_verified", pd.Series(False, index=head_rows.index)).map(as_bool).sum())
            torso_excluded = int(torso_rows.exclude.map(lambda value: str(value).strip().casefold() in
                                 {"1", "true", "yes", "y", "excluded"}).sum()) if "exclude" in torso_rows else 0
            lines.append(f"{name}\n  torso: {len(torso_rows)}（排除 {torso_excluded}）"
                         f"    Head: {head_present}（确认 {head_verified}）"
                         f"    Reflection: {reflection_present}（确认 {reflection_verified}）")
        messagebox.showinfo("现有标注统计", "\n\n".join(lines) if lines else "没有标注记录")

    def view_annotations(self, source: str = "all") -> None:
        """Open the paginated contact sheet without blocking this GUI."""
        if not self._sync_args():
            return
        if not (self.args.dataset / "images").is_dir() and not self.args.labels.is_file():
            messagebox.showerror("没有标注", "未找到训练数据集，也未找到多边形标注 CSV。")
            return
        argv = [sys.executable, str(R.UNET / "view_annotations.py"),
                "--dataset", str(self.args.dataset), "--labels", str(self.args.labels),
                "--heads", str(self.args.heads),
                "--arena-width-cm", str(self.args.arena_width_cm),
                "--arena-height-cm", str(self.args.arena_height_cm),
                "--source", source]
        for cfg in self.videos:
            video = Path(cfg["video"])
            if video.is_file():
                argv.extend(["--video", str(video), "--roi-json", str(self.roi_path(cfg))])
        try:
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            subprocess.Popen(argv, creationflags=flags)
            self.log(f"\n$ {subprocess.list2cmdline(argv)}\n")
            self.status_var.set("已打开分页标注预览")
        except OSError as exc:
            messagebox.showerror("无法打开标注预览", str(exc))

    # ---------------------------------------------------------- task runner
    def run_commands(self, commands: list[tuple[Path, list[str]]], label: str) -> None:
        if self.running:
            messagebox.showwarning("任务运行中", "请等待当前任务结束，或点击停止。"); return
        self._save_state(); self.cancelled = False
        self.running = True
        self.command_queue = deque(commands)
        self.status_var.set(f"{label}：准备运行 {len(commands)} 个任务")
        self.progress.start(10); self._set_busy(True)
        self._launch_next()

    def _launch_next(self) -> None:
        if self.cancelled:
            self._finish(False, "已停止")
            return
        if not self.command_queue:
            self._finish(True, "完成")
            return
        script, argv = self.command_queue.popleft()
        self.current_script = script
        self.status_var.set(f"运行：{script.name}（剩余 {len(self.command_queue)}）")
        pretty = subprocess.list2cmdline([sys.executable, str(script), *argv])
        self.log(f"\n$ {pretty}\n")

        def worker():
            try:
                if self.cancelled:
                    self.events.put(("done", -15)); return
                flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                proc = subprocess.Popen(
                    [sys.executable, "-u", str(script), *argv], cwd=str(R.ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    creationflags=flags)
                self.proc = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.events.put(("log", line))
                self.events.put(("done", proc.wait()))
            except Exception as exc:  # noqa: BLE001
                self.events.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                    self._consume_training_json(payload)
                elif kind == "done":
                    self.proc = None
                    self.log(f"--- 退出码 {payload} ---\n")
                    if self.cancelled:
                        self._finish(False, "已停止")
                    elif payload == 1 and self.current_script is not None \
                            and self.current_script.name == "calibrate_model.py":
                        self.command_queue.clear()
                        self._finish(True, "没有实际修改，已跳过重训")
                    elif payload == 1 and self.current_script is not None \
                            and self.current_script.name == "incremental_review.py":
                        self.command_queue.clear()
                        self._finish(True, "已取消增量训练，没有修改模型")
                    elif payload == 0:
                        self._launch_next()
                    else:
                        self.command_queue.clear(); self._finish(False, f"失败（退出码 {payload}）")
                elif kind == "error":
                    self.proc = None; self.command_queue.clear()
                    self.log(f"ERROR: {payload}\n"); self._finish(False, "启动失败")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _finish(self, success: bool, text: str) -> None:
        self.command_queue.clear(); self.progress.stop(); self.running = False
        self.current_script = None; self._set_busy(False)
        self.status_var.set(text); self.refresh(); self._save_state()
        if not success and not self.cancelled:
            messagebox.showerror("任务失败", "请查看下方日志中的最后一个 ERROR/Traceback。")

    def stop(self) -> None:
        self.cancelled = True; self.command_queue.clear()
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                self.log("\n--- 已请求停止当前子进程 ---\n")
            except OSError as exc:
                self.log(f"停止失败: {exc}\n")
        elif not self.running:
            self._finish(False, "已停止")
        else:
            self.status_var.set("正在停止启动中的任务…")

    def _set_busy(self, busy: bool) -> None:
        for button in self.action_buttons:
            button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")

    # ----------------------------------------------------------- video/path
    def add_video(self) -> None:
        filename = filedialog.askopenfilename(title="添加视频", initialdir=str(R.ROOT / "data"),
                                              filetypes=[("视频", "*.avi *.mp4 *.mov *.mkv")])
        if not filename: return
        video = Path(filename)
        for index, cfg in enumerate(self.videos):
            if self._same_path(cfg["video"], video):
                self.current = index; self._fill_video_combo(); self._select_current(); return
        self.videos.append({"video": video, "roi": Path(""), "compare_out": Path(""),
                            "output": R.ROOT / "results" / f"{video.stem}_unet"})
        self.current = len(self.videos) - 1
        self._fill_video_combo(); self._select_current(); self._save_state()

    def remove_video(self) -> None:
        if len(self.videos) <= 1:
            messagebox.showinfo("不能移除", "列表中至少保留一个视频。"); return
        if not messagebox.askyesno("移除视频", "仅从 UI 列表移除，不删除原视频和结果。继续吗？"):
            return
        self.videos.pop(self.current); self.current = min(self.current, len(self.videos) - 1)
        self._fill_video_combo(); self._select_current(); self._save_state()

    def choose_roi_file(self) -> None:
        filename = filedialog.askopenfilename(title="选择 ROI JSON", initialdir=str(R.ROOT / "traditional" / "basic_rois"),
                                              filetypes=[("JSON", "*.json")])
        if filename:
            self.cfg()["roi"] = Path(filename); self.refresh(); self._save_state()

    def choose_model(self) -> None:
        filename = filedialog.askopenfilename(title="选择模型", initialdir=str(self.args.model.parent),
                                              filetypes=[("PyTorch checkpoint", "*.pt")])
        if filename:
            self.model_var.set(filename); self._sync_args(False); self.refresh(); self._save_state()

    def choose_output(self) -> None:
        directory = filedialog.askdirectory(title="选择当前视频输出目录", initialdir=str(self.output_path()))
        if directory:
            self.output_var.set(directory); self.cfg()["output"] = Path(directory)
            self._sync_args(False); self.refresh(); self._save_state()

    def choose_dataset(self) -> None:
        directory = filedialog.askdirectory(title="选择训练数据集目录", initialdir=str(self.args.dataset))
        if directory:
            self.dataset_var.set(directory); self._sync_args(False); self.refresh(); self._save_state()

    def open_path(self, path: Path) -> None:
        target = path if path.exists() else path.parent
        if not target.exists():
            messagebox.showinfo("尚无产物", str(path)); return
        try:
            os.startfile(str(target))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("无法打开", str(exc))

    def _fill_video_combo(self) -> None:
        names = [Path(v["video"]).name for v in self.videos]
        self.video_combo["values"] = names
        self.video_combo.current(self.current)
        self.video_var.set(names[self.current])

    def _select_current(self, _event=None) -> None:
        selected = self.video_combo.current()
        if selected >= 0: self.current = selected
        self.output_var.set(str(self.output_path()))
        self.args.infer_out = self.output_path()
        self._sync_args(show_error=False)
        self.refresh(); self._save_state()

    # --------------------------------------------------------------- logging
    def log(self, text: str) -> None:
        self.events.put(("log", text))

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text); self.log_lines += text.count("\n")
        if self.log_lines > LOG_MAX_LINES:
            excess = self.log_lines - LOG_MAX_LINES
            self.log_text.delete("1.0", f"{excess + 1}.0"); self.log_lines -= excess
        self.log_text.see(tk.END)

    def _consume_training_json(self, line: str) -> None:
        if line.startswith("MODEL_OUTPUT="):
            model = Path(line.split("=", 1)[1].strip())
            self.model_var.set(str(model)); self.args.model = model
            self.log(f"GUI 已自动选择新模型：{model.name}\n")
            return
        if '"epoch"' not in line or '"train_loss"' not in line: return
        try: row = json.loads(line.strip())
        except json.JSONDecodeError: return
        self.train_curve[int(row["epoch"])] = (float(row["train_loss"]), float(row["val_dice"]))
        self.draw_curve()

    def draw_curve(self) -> None:
        cv = self.curve; cv.delete("all"); width = max(cv.winfo_width(), 400); height = 145
        if not self.train_curve:
            cv.create_text(width / 2, height / 2, text="训练时实时显示 loss / validation Dice",
                           fill="#777", font=SMALL); return
        epochs = sorted(self.train_curve); values = [self.train_curve[e] for e in epochs]
        for panel, index, title, color in ((0, 0, "Loss", "#1f5ea8"), (1, 1, "Val Dice", "#18864b")):
            y0 = 5 + panel * 70; y1 = y0 + 62
            cv.create_rectangle(42, y0, width - 8, y1, outline="#d0d6dc")
            series = [v[index] for v in values]; lo, hi = min(series), max(series)
            if hi - lo < 1e-9: hi = lo + 1
            points = []
            for j, value in enumerate(series):
                x = 44 + j / max(len(series) - 1, 1) * (width - 55)
                y = y1 - 5 - (value - lo) / (hi - lo) * 45
                points += [x, y]
            if len(points) >= 4: cv.create_line(*points, fill=color, width=2)
            cv.create_text(5, y0 + 4, anchor="nw", text=title, fill=color, font=SMALL)
            cv.create_text(width - 12, y0 + 4, anchor="ne", text=f"epoch {epochs[-1]}", fill="#666", font=SMALL)

    # ------------------------------------------------------------------- UI
    def _button(self, parent, text, command, **kwargs):
        button = ttk.Button(parent, text=text, command=command, **kwargs)
        self.action_buttons.append(button); return button

    def _path_row(self, parent, label, variable, choose, open_command=None):
        row = ttk.Frame(parent); row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=12, font=FONT).pack(side="left")
        ttk.Entry(row, textvariable=variable, font=SMALL).pack(side="left", fill="x", expand=True, padx=4)
        self._button(row, "选择", choose, width=7).pack(side="left")
        if open_command: self._button(row, "打开", open_command, width=7).pack(side="left", padx=(4, 0))

    def _build_ui(self) -> None:
        root = self.root; root.geometry("1120x860"); root.minsize(960, 720)
        style = ttk.Style(); style.configure("Primary.TButton", font=("Microsoft YaHei UI", 11, "bold"))

        header = ttk.Frame(root); header.pack(fill="x", padx=10, pady=(8, 4))
        ttk.Label(header, text="Behavior Analyze · Fibre-aware U-Net v3", font=("Microsoft YaHei UI", 15, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status_var, foreground="#165d9c", font=FONT).pack(side="left", padx=18)
        self.progress = ttk.Progressbar(header, mode="indeterminate", length=180); self.progress.pack(side="left", padx=6)
        self.stop_button = ttk.Button(header, text="停止任务", command=self.stop, state="disabled"); self.stop_button.pack(side="right")

        bar = ttk.LabelFrame(root, text="项目视频"); bar.pack(fill="x", padx=10, pady=4)
        self.video_combo = ttk.Combobox(bar, textvariable=self.video_var, state="readonly", width=55, font=FONT)
        self.video_combo.pack(side="left", padx=6, pady=6); self.video_combo.bind("<<ComboboxSelected>>", self._select_current)
        self._button(bar, "添加视频", self.add_video).pack(side="left", padx=3)
        self._button(bar, "移出列表", self.remove_video).pack(side="left", padx=3)
        ttk.Label(bar, textvariable=self.roi_summary_var, foreground="#666", font=SMALL).pack(side="left", padx=10)

        tabs = ttk.Notebook(root); tabs.pack(fill="both", expand=True, padx=10, pady=4)
        analyze_tab = ttk.Frame(tabs); train_tab = ttk.Frame(tabs); annotation_tab = ttk.Frame(tabs)
        tabs.add(analyze_tab, text="  一键分析  "); tabs.add(train_tab, text="  使用已有标注训练  "); tabs.add(annotation_tab, text="  高级标注工具  ")

        paths = ttk.LabelFrame(analyze_tab, text="当前分析项目"); paths.pack(fill="x", padx=8, pady=8)
        roi_row = ttk.Frame(paths); roi_row.pack(fill="x", pady=3)
        ttk.Label(roi_row, text="竞技场 ROI", width=12, font=FONT).pack(side="left")
        ttk.Label(roi_row, textvariable=self.roi_summary_var, font=SMALL, foreground="#555").pack(side="left", fill="x", expand=True)
        self._button(roi_row, "框选 ROI", lambda: self.advanced_step("roi")).pack(side="right", padx=3)
        self._button(roi_row, "选择 JSON", self.choose_roi_file).pack(side="right", padx=3)
        self._path_row(paths, "模型", self.model_var, self.choose_model,
                       lambda: self.open_path(Path(self.model_var.get()).parent))
        self._path_row(paths, "输出目录", self.output_var, self.choose_output,
                       lambda: self.open_path(Path(self.output_var.get())))

        params = ttk.LabelFrame(analyze_tab, text="光纤与时间约束参数"); params.pack(fill="x", padx=8, pady=4)
        for text, variable, width in (("概率阈值", self.threshold_var, 7), ("光纤开运算", self.fibre_var, 5),
                                      ("重捕获秒数", self.reacquire_var, 6)):
            ttk.Label(params, text=text, font=FONT).pack(side="left", padx=(8, 2), pady=7)
            ttk.Entry(params, textvariable=variable, width=width, font=FONT).pack(side="left")
        ttk.Label(params, text="画面旋转", font=FONT).pack(side="left", padx=(12, 2))
        ttk.Combobox(params, textvariable=self.rotate_var, values=[0, 90, 180, 270], state="readonly", width=5).pack(side="left")
        ttk.Label(params, text="建议：阈值 0.5；细光纤 5，粗光纤 7；重捕获 0.35 s",
                  font=SMALL, foreground="#666").pack(side="left", padx=12)

        action = ttk.Frame(analyze_tab); action.pack(fill="x", padx=8, pady=10)
        self._button(action, "一键完整分析", self.analyze, style="Primary.TButton", width=18).pack(side="left")
        self._button(action, "仅导出分割预览", self.mask_preview, width=16).pack(side="left", padx=8)
        self._button(action, "打开 Mask 视频", lambda: self.open_path(self.output_path() / "mouse_miniscope_mask.mp4")).pack(side="left", padx=3)
        self._button(action, "打开结果视频", lambda: self.open_path(self.output_path() / "head_track_overlay.mp4")).pack(side="left", padx=3)
        self._button(action, "加入训练并继续微调", self.incremental_train,
                     width=20).pack(side="left", padx=8)
        ttk.Label(analyze_tab, textvariable=self.analysis_summary_var, font=FONT, foreground="#176b3a").pack(anchor="w", padx=12)
        ttk.Label(analyze_tab, text="完整分析只解码视频一次，同时输出 mask、头/身体叠加视频、轨迹 CSV 和元数据。",
                  font=SMALL, foreground="#555").pack(anchor="w", padx=12, pady=5)

        quick_train = ttk.LabelFrame(train_tab, text="常用入口 · 直接复用全部已有标注")
        quick_train.pack(fill="x", padx=8, pady=8)
        ttk.Label(quick_train,
                  text="全量同步 CSV → 清理失效样本 → 热启动旧模型 → 训练并验证；不需要重新逐帧打标。",
                  font=FONT, foreground="#263746").pack(anchor="w", padx=10, pady=(8, 5))
        quick_actions = ttk.Frame(quick_train); quick_actions.pack(fill="x", padx=10, pady=(0, 10))
        self._button(quick_actions, "一键重建数据集并训练", lambda: self.rebuild_dataset(True),
                     style="Primary.TButton", width=24).pack(side="left")
        self._button(quick_actions, "拼接查看全部已有标注", lambda: self.view_annotations("all"),
                     width=22).pack(side="left", padx=8)
        ttk.Label(quick_actions, text="每页 20 张；←/→ 翻页", font=SMALL,
                  foreground="#666").pack(side="left", padx=4)

        train_info = ttk.LabelFrame(train_tab, text="训练设置"); train_info.pack(fill="x", padx=8, pady=4)
        ttk.Label(train_info, text="程序会从 CSV 全量同步数据集、清除旧样本、复用旧模型权重，并仅在验证不退化时晋升候选模型。",
                  font=SMALL, foreground="#444").pack(anchor="w", padx=8, pady=6)
        self._path_row(train_info, "数据集目录", self.dataset_var, self.choose_dataset,
                       lambda: self.open_path(Path(self.dataset_var.get())))
        train_set = ttk.Frame(train_info); train_set.pack(fill="x", padx=6, pady=5)
        for text, variable, width in (("Epoch", self.epochs_var, 6), ("Batch", self.batch_var, 5),
                                      ("学习率", self.lr_var, 8)):
            ttk.Label(train_set, text=text, font=FONT).pack(side="left", padx=(6, 2))
            ttk.Entry(train_set, textvariable=variable, width=width).pack(side="left")
        ttk.Checkbutton(train_set, text="使用列表中全部视频", variable=self.all_train_var).pack(side="left", padx=14)
        train_actions = ttk.Frame(train_tab); train_actions.pack(fill="x", padx=8, pady=4)
        self._button(train_actions, "仅重建数据集", self.rebuild_dataset, width=15).pack(side="left")
        self._button(train_actions, "仅训练", self.train, width=12).pack(side="left", padx=6)
        self._button(train_actions, "查看上次重建样本", lambda: self.view_annotations("dataset"), width=18).pack(side="left")
        ttk.Label(train_tab, textvariable=self.training_summary_var, font=FONT, foreground="#176b3a").pack(anchor="w", padx=12, pady=5)
        self.curve = tk.Canvas(train_tab, height=145, bg="#fafbfc", highlightthickness=1,
                               highlightbackground="#ccd3da"); self.curve.pack(fill="x", padx=8, pady=5)
        self.curve.bind("<Configure>", lambda _e: self.draw_curve())

        warning = ttk.LabelFrame(annotation_tab, text="高级工具：只有出现新的失败类型时才使用")
        warning.pack(fill="x", padx=8, pady=8)
        ttk.Label(warning, text="不要把“校准并重训”当成常规循环。已校准帧会跳过，未实际编辑则不会触发重训。",
                  foreground="#9a4d00", font=FONT).pack(anchor="w", padx=8, pady=6)
        advanced = ttk.Frame(annotation_tab); advanced.pack(fill="x", padx=8, pady=6)
        self._button(advanced, "生成传统对比数据", lambda: self.advanced_step("compare")).pack(side="left", padx=3)
        self._button(advanced, "筛选新帧", lambda: self.advanced_step("screen")).pack(side="left", padx=3)
        self._button(advanced, "补充多边形标注", lambda: self.advanced_step("annotate")).pack(side="left", padx=3)
        corrections = ttk.Frame(annotation_tab); corrections.pack(fill="x", padx=8, pady=(0, 6))
        self._button(corrections, "复查/修正已有轮廓", lambda: self.advanced_step("review")).pack(side="left", padx=3)
        self._button(corrections, "根据结果补充 Head 标记", lambda: self.advanced_step("head_result")).pack(side="left", padx=3)
        self._button(corrections, "校准并重训", lambda: self.advanced_step("calibrate")).pack(side="left", padx=3)
        self._button(corrections, "查看标注统计", self.show_annotation_stats).pack(side="left", padx=3)
        ttk.Label(annotation_tab, text="每视频最多新增帧数", font=FONT).pack(anchor="w", padx=12, pady=(10, 2))
        ttk.Spinbox(annotation_tab, from_=1, to=100, textvariable=self.per_video_var, width=6).pack(anchor="w", padx=12)

        log_frame = ttk.LabelFrame(root, text="任务日志"); log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        log_bar = ttk.Frame(log_frame); log_bar.pack(fill="x")
        ttk.Button(log_bar, text="清空日志", command=lambda: (self.log_text.delete("1.0", tk.END), setattr(self, "log_lines", 0))).pack(side="right", padx=4)
        self.log_text = tk.Text(log_frame, height=12, font=("Consolas", 9), bg="#101419", fg="#d7e0e7", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.draw_curve()

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("任务仍在运行", "关闭会终止当前任务。确定关闭吗？"):
                return
            self.stop()
        self._save_state(); self.root.destroy()


def main() -> None:
    root = tk.Tk(); App(root); root.mainloop()


if __name__ == "__main__":
    main()
