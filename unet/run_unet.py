#!/usr/bin/env python3
"""One-command launcher for the U-Net segmentation workflow.

Modes
-----
check     print what exists and what is still missing
roi       interactive arena-corner picker for a NEW video (make_roi.py)
compare   (re)generate head_method_comparison.csv for every video
screen    pick maximally diverse frames + manually exclude junk frames
          (mouse absent, human intervention, ...) -> screening.csv
annotate  open the torso-polygon labelling window on the screened frames
          (8-point polygons; already labelled frames are skipped)
prepare   export polygon labels from every video into one U-Net dataset
train     train the U-Net (best validation Dice checkpoint)
calibrate manual model calibration: 20 least-confident frames, correct
          polygon + head + reflection, then auto re-prepare + retrain
infer     segment a new video; excluded frames -> NaN + EXCLUDED overlay
head      body (U-Net mask centroid) + head (miniscope reflection) tracking
ui        GUI wizard: training / processing tabs, one click per step

Loop modes (compare/screen/annotate/prepare/roi) run over the VIDEOS list
below. Single-video modes (train/infer/head) use the first video; override
with --video. Add --interactive to pick files with dialogs (train/infer/head:
head also asks for the ROI JSON).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repository root
UNET = ROOT / "unet"
CODE = ROOT / "traditional" / "code"

# ---------------------------------------------------------------------------
# DEFAULTS - edit this block once for your project, or override per run.
# ---------------------------------------------------------------------------
VIDEOS = [
    {
        "video": ROOT / "data" / "12026-08-12 160535.avi",
        "roi": ROOT / "traditional" / "basic_rois" / "160535_roi.json",
        "compare_out": ROOT / "traditional" / "results" / "basic_recognition" / "160535",
    },
    {
        "video": ROOT / "data" / "12026-08-12 161211.avi",
        "roi": ROOT / "traditional" / "basic_rois" / "161211_roi.json",
        "compare_out": ROOT / "traditional" / "results" / "basic_recognition" / "161211",
    },
    {
        "video": ROOT / "data" / "HQ312-freelymoving_2026-08-12_15-44-14" / "MiceVideo1" / "MiceVideo" / "12026-08-12 154415.avi",
        "roi": ROOT / "traditional" / "basic_rois" / "154415_roi.json",
        "compare_out": ROOT / "traditional" / "results" / "basic_recognition" / "154415",
    },
]
SCREENING = ROOT / "traditional" / "results" / "screening.csv"
LABELS = ROOT / "traditional" / "results" / "manual_torso_constraints.csv"
HEADS = ROOT / "traditional" / "results" / "head_anchor_calibration.csv"
DATASET = ROOT / "unet" / "datasets" / "project"
MODEL = ROOT / "unet" / "models" / "project" / "best_unet.pt"
INFER_OUT = ROOT / "results" / "video_unet"
ARENA_WIDTH_CM = 25.0
ARENA_HEIGHT_CM = 30.0


def pick_file(title: str, initial: Path, pattern: str) -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(title=title, initialdir=str(initial), filetypes=[("files", pattern)])
    root.destroy()
    return path


def pick_dir(title: str, initial: Path) -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title=title, initialdir=str(initial))
    root.destroy()
    return path


def pick_or_create_roi(video: Path) -> Path:
    """ROI for the picked video; auto-opens the corner picker when a new
    recording has none yet, so head --interactive needs no separate roi
    step. Lookup order: default {stem}_roi.json -> any basic_rois JSON whose
    "input" field points at this video -> interactive corner picker.
    On cancel/error falls back to picking an existing JSON."""
    rois = ROOT / "traditional" / "basic_rois"
    default = rois / f"{video.stem}_roi.json"
    if default.is_file():
        return default
    for f in sorted(rois.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if str(data.get("input", "")).replace("\\", "/").endswith(video.name):
                return f
        except (json.JSONDecodeError, OSError):
            continue
    print(f"No ROI JSON for {video.name}; opening the corner picker "
          "(click the 4 arena corners, S to save, Q to cancel)...")
    code = run(UNET / "make_roi.py", ["--video", str(video)])
    if not code and default.is_file():
        return default
    picked = pick_file("Select ROI JSON (arena corners)", default.parent, "*.json")
    return Path(picked) if picked else default


def apply_dialogs(v: dict, mode: str) -> dict:
    if mode == "train":
        v["dataset"] = Path(pick_dir("Select dataset dir (with images/ and masks/)", v["dataset"]) or v["dataset"])
    if mode in ("infer", "head"):
        v["video"] = Path(pick_file("Select source video", v["video"].parent, "*.avi *.mp4 *.mov *.mkv") or v["video"])
        v["model"] = Path(pick_file("Select best_unet.pt", v["model"].parent, "*.pt") or v["model"])
        v["infer_out"] = Path(pick_dir("Select inference output dir", v["infer_out"]) or v["infer_out"])
    if mode == "head":
        v["roi"] = pick_or_create_roi(v["video"])
    return v


def compare_cmd(cfg: dict, w: str, h: str) -> tuple[Path, list[str]]:
    return CODE / "compare_head_methods.py", [
        "--input", str(cfg["video"]), "--output-dir", str(cfg["compare_out"]),
        "--roi-json", str(cfg["roi"]), "--arena-width-cm", w, "--arena-height-cm", h]


def screen_cmd(cfg: dict, w: str, h: str, a) -> tuple[Path, list[str]]:
    return UNET / "screen_frames.py", [
        "--video", str(cfg["video"]), "--roi", str(cfg["roi"]), "--output", str(a.screening),
        "--per-video", str(a.per_video), "--junk", str(a.junk),
        "--arena-width-cm", w, "--arena-height-cm", h]


def annotate_cmd(cfg: dict, w: str, h: str, a) -> tuple[Path, list[str]]:
    return CODE / "annotate_torso_constraints.py", [
        "--input", str(cfg["video"]),
        "--comparison-csv", str(cfg["compare_out"] / "head_method_comparison.csv"),
        "--roi-json", str(cfg["roi"]), "--output", str(a.labels),
        "--arena-width-cm", w, "--arena-height-cm", h,
        "--max-labels", str(a.max_labels), "--candidate-csv", str(a.screening)]


def prepare_cmd(cfg: dict, w: str, h: str, a) -> tuple[Path, list[str]]:
    return UNET / "prepare_dataset.py", [
        "--video", str(cfg["video"]), "--labels", str(a.labels),
        "--output-dir", str(a.dataset), "--size", str(a.size),
        "--exclude-csv", str(a.screening)]


def train_cmd(v: dict, a) -> tuple[Path, list[str]]:
    return UNET / "train.py", [
        "--dataset", str(v["dataset"]), "--output-dir", str(v["model"].parent),
        "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
        "--lr", f"{a.lr:g}"]


def calibrate_cmd(v: dict, cfg: dict, a) -> tuple[Path, list[str]]:
    return UNET / "calibrate_model.py", [
        "--video", str(cfg["video"]), "--model", str(v["model"]),
        "--roi-json", str(cfg["roi"]), "--labels", str(v["labels"]),
        "--heads", str(v["heads"]),
        "--comparison-csv", str(cfg["compare_out"] / "head_method_comparison.csv"),
        "--exclude-csv", str(a.screening), "--n-frames", str(a.calib_frames),
        "--arena-width-cm", f"{a.arena_width_cm:.2f}", "--arena-height-cm", f"{a.arena_height_cm:.2f}",
        "--threshold", f"{a.threshold:.2f}", "--rotate", str(a.rotate)]


def infer_cmd(v: dict, a) -> tuple[Path, list[str]]:
    return UNET / "infer.py", [
        "--video", str(v["video"]), "--model", str(v["model"]),
        "--output-dir", str(v["infer_out"]), "--threshold", f"{a.threshold:.2f}",
        "--rotate", str(a.rotate), "--exclude-csv", str(a.screening)]


def head_cmd(v: dict, cfg: dict, a) -> tuple[Path, list[str]]:
    return UNET / "head_track.py", [
        "--video", str(cfg["video"]), "--model", str(v["model"]),
        "--roi-json", str(cfg["roi"]), "--output-dir", str(v["infer_out"]),
        "--arena-width-cm", f"{a.arena_width_cm:.2f}", "--arena-height-cm", f"{a.arena_height_cm:.2f}",
        "--threshold", f"{a.threshold:.2f}", "--rotate", str(a.rotate),
        "--exclude-csv", str(a.screening)]


def roi_cmd(v: dict) -> tuple[Path, list[str]]:
    return UNET / "make_roi.py", ["--video", str(v["video"])]


def run(script: Path, argv: list[str]) -> int:
    print(f"$ python {script.relative_to(ROOT)} {' '.join(argv)}")
    return subprocess.run([sys.executable, str(script), *argv]).returncode


def check(a) -> None:
    rows = []
    for cfg in VIDEOS:
        rows.append((cfg["video"].name, "video", cfg["video"].is_file()))
        rows.append((cfg["video"].name, "roi", cfg["roi"].is_file()))
        rows.append((cfg["video"].name, "comparison.csv", (cfg["compare_out"] / "head_method_comparison.csv").is_file()))
    dataset_ok = (a.dataset / "images").is_dir() and any((a.dataset / "images").glob("*.png"))
    rows += [
        ("screening", "screening.csv", a.screening.is_file()),
        ("labels", "manual_torso_constraints.csv", a.labels.is_file()),
        ("dataset", "unet/datasets/project", dataset_ok),
        ("model", "best_unet.pt", a.model.is_file()),
    ]
    torch_ok = _torch_ok()
    print(f'{"Video/Item":<34}{"Artifact":<22}{"Status":<9}Path')
    for video, artifact, ok in rows:
        print(f'{video:<34}{artifact:<22}{"OK" if ok else "MISSING":<9}')
    print(f'{"":<34}{"PyTorch+CUDA":<22}{"OK" if torch_ok else "MISSING":<9}')
    missing = [row[1] for row in rows if not row[2]]
    if not torch_ok:
        missing.append("PyTorch")
    print()
    if missing:
        print("Missing: " + ", ".join(missing))
        print("Order: pip install torch -> compare -> screen -> annotate -> prepare -> train -> infer")
    else:
        print("Everything in place - run: python unet/run_unet.py infer")


def _torch_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["check", "compare", "screen", "annotate", "prepare", "train", "calibrate", "infer", "head", "roi", "ui"])
    p.add_argument("--interactive", action="store_true", help="pick files with dialogs (train/infer)")
    p.add_argument("--video", type=Path, help="single-video override for loop modes")
    p.add_argument("--roi", type=Path); p.add_argument("--compare-out", type=Path, dest="compare_out")
    p.add_argument("--labels", type=Path, default=LABELS)
    p.add_argument("--heads", type=Path, default=HEADS)
    p.add_argument("--screening", type=Path, default=SCREENING)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--infer-out", type=Path, dest="infer_out", default=INFER_OUT)
    p.add_argument("--arena-width-cm", type=float, default=ARENA_WIDTH_CM)
    p.add_argument("--arena-height-cm", type=float, default=ARENA_HEIGHT_CM)
    p.add_argument("--max-labels", type=int, default=100)
    p.add_argument("--per-video", type=int, default=40, help="screened candidates per video")
    p.add_argument("--junk", type=int, default=20, help="junk frames shown per video")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--calib-frames", type=int, default=20,
                   help="low-confidence frames to calibrate after training")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="arena turned vs training: rotate input frames for infer/head")
    a = p.parse_args()

    videos = list(VIDEOS)
    if a.video:
        cfg = {"video": a.video, "roi": a.roi or Path(""), "compare_out": a.compare_out or Path("")}
        if not cfg["roi"].is_file() or not cfg["compare_out"].is_dir():
            p.error("--video override needs --roi and --compare-out")
        videos = [cfg]
    v = {"video": videos[0]["video"], "model": a.model, "infer_out": a.infer_out,
         "dataset": a.dataset, "labels": a.labels, "screening": a.screening,
         "heads": a.heads, "roi": videos[0]["roi"],
         "compare_out": videos[0]["compare_out"]}
    if a.interactive:
        v = apply_dialogs(v, a.mode)

    w = f"{a.arena_width_cm:.2f}"; h = f"{a.arena_height_cm:.2f}"
    if a.mode == "check":
        check(a); return
    code = 0
    if a.mode == "compare":
        for cfg in videos:
            code = run(*compare_cmd(cfg, w, h)) or code
    elif a.mode == "screen":
        for cfg in videos:
            code = run(*screen_cmd(cfg, w, h, a)) or code
    elif a.mode == "annotate":
        for cfg in videos:
            code = run(*annotate_cmd(cfg, w, h, a)) or code
    elif a.mode == "prepare":
        for cfg in videos:
            code = run(*prepare_cmd(cfg, w, h, a)) or code
    elif a.mode == "train":
        code = run(*train_cmd(v, a))
    elif a.mode == "calibrate":
        cfg = {"video": v["video"], "roi": v["roi"], "compare_out": v["compare_out"]}
        code = run(*calibrate_cmd(v, cfg, a))
        if code == 0:  # corrections saved -> re-export + retrain
            for cfg in videos:
                code = run(*prepare_cmd(cfg, w, h, a)) or code
            code = run(*train_cmd(v, a)) or code
    elif a.mode == "infer":
        cfg = {"video": v["video"]}
        code = run(*infer_cmd(v, a))
    elif a.mode == "head":
        cfg = {"video": v["video"], "roi": v["roi"]}
        code = run(*head_cmd(v, cfg, a))
    elif a.mode == "roi":
        for cfg in videos:
            code = run(*roi_cmd(cfg)) or code
    elif a.mode == "ui":
        from ui import main as ui_main
        ui_main()
        return
    raise SystemExit(code)


if __name__ == "__main__":
    main()