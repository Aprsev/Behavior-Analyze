#!/usr/bin/env python3
"""One-command launcher for the U-Net segmentation workflow.

Modes
-----
check     print what exists and what is still missing
compare   (re)generate head_method_comparison.csv used by the annotator
annotate  open the interactive torso-polygon labelling window
prepare   export polygon labels to the U-Net image/mask dataset
train     train the U-Net (best validation Dice checkpoint)
infer     segment a new video; write mask/overlay MP4s + trajectory CSV

Every mode runs with the defaults below; override any path with the
matching --flag. Add --interactive to pick files with dialogs instead.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repository root
UNET = ROOT / "unet"
CODE = ROOT / "traditional" / "code"

# ---------------------------------------------------------------------------
# DEFAULTS - edit this block once for your project, or override per run.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "video":           ROOT / "data" / "12026-08-12 160535.avi",
    "roi":             ROOT / "traditional" / "basic_rois" / "160535_roi.json",
    "compare_out":     ROOT / "traditional" / "results" / "basic_recognition" / "160535",
    "labels":          ROOT / "traditional" / "results" / "manual_torso_constraints.csv",
    "dataset":         ROOT / "unet" / "datasets" / "project",
    "model":           ROOT / "unet" / "models" / "project" / "best_unet.pt",
    "infer_out":       ROOT / "results" / "video_unet",
    "arena_width_cm":  25.0,
    "arena_height_cm": 30.0,
}


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


def apply_dialogs(mode: str, v: dict) -> dict:
    """Ask for the mode's key inputs with file dialogs (tkinter, no install)."""
    if mode in ("compare", "annotate"):
        v["video"] = Path(pick_file("Select source video", v["video"].parent, "*.avi *.mp4 *.mov *.mkv") or v["video"])
        v["roi"] = Path(pick_file("Select ROI JSON", v["roi"].parent, "*.json") or v["roi"])
    if mode == "annotate":
        v["compare_out"] = Path(pick_file("Select head_method_comparison.csv", v["compare_out"], "*.csv") or v["compare_out"]).parent
        v["labels"] = Path(pick_file("Select label CSV output (may be new)", v["labels"].parent, "*.csv") or v["labels"])
    if mode == "prepare":
        v["video"] = Path(pick_file("Select source video", v["video"].parent, "*.avi *.mp4 *.mov *.mkv") or v["video"])
        v["labels"] = Path(pick_file("Select manual_torso_constraints.csv", v["labels"].parent, "*.csv") or v["labels"])
        v["dataset"] = Path(pick_dir("Select dataset output dir", v["dataset"]) or v["dataset"])
    if mode == "train":
        v["dataset"] = Path(pick_dir("Select dataset dir (with images/ and masks/)", v["dataset"]) or v["dataset"])
    if mode == "infer":
        v["video"] = Path(pick_file("Select source video", v["video"].parent, "*.avi *.mp4 *.mov *.mkv") or v["video"])
        v["model"] = Path(pick_file("Select best_unet.pt", v["model"].parent, "*.pt") or v["model"])
        v["infer_out"] = Path(pick_dir("Select inference output dir", v["infer_out"]) or v["infer_out"])
    return v


def command(mode: str, v: dict, args) -> tuple[Path, list[str]]:
    w = f'{v["arena_width_cm"]:.2f}'; h = f'{v["arena_height_cm"]:.2f}'
    if mode == "compare":
        return CODE / "compare_head_methods.py", [
            "--input", str(v["video"]), "--output-dir", str(v["compare_out"]),
            "--roi-json", str(v["roi"]), "--arena-width-cm", w, "--arena-height-cm", h]
    if mode == "annotate":
        return CODE / "annotate_torso_constraints.py", [
            "--input", str(v["video"]),
            "--comparison-csv", str(v["compare_out"] / "head_method_comparison.csv"),
            "--roi-json", str(v["roi"]), "--output", str(v["labels"]),
            "--arena-width-cm", w, "--arena-height-cm", h, "--max-labels", str(args.max_labels)]
    if mode == "prepare":
        return UNET / "prepare_dataset.py", [
            "--video", str(v["video"]), "--labels", str(v["labels"]),
            "--output-dir", str(v["dataset"]), "--size", str(args.size)]
    if mode == "train":
        return UNET / "train.py", [
            "--dataset", str(v["dataset"]), "--output-dir", str(v["model"].parent),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size)]
    if mode == "infer":
        return UNET / "infer.py", [
            "--video", str(v["video"]), "--model", str(v["model"]),
            "--output-dir", str(v["infer_out"]), "--threshold", f'{args.threshold:.2f}']
    raise ValueError(f"unknown mode: {mode}")


def check(v: dict) -> None:
    dataset_ok = (v["dataset"] / "images").is_dir() and any((v["dataset"] / "images").glob("*.png"))
    rows = [
        ("Source video", v["video"], v["video"].is_file()),
        ("ROI JSON", v["roi"], v["roi"].is_file()),
        ("comparison CSV", v["compare_out"] / "head_method_comparison.csv", (v["compare_out"] / "head_method_comparison.csv").is_file()),
        ("torso labels", v["labels"], v["labels"].is_file()),
        ("UNet dataset", v["dataset"], dataset_ok),
        ("trained model", v["model"], v["model"].is_file()),
    ]
    torch_ok = _torch_ok()
    print(f'{"Item":<16}{"Status":<9}Path')
    for name, path, ok in rows:
        print(f'{name:<16}{"OK" if ok else "MISSING":<9}{path}')
    print(f'{"PyTorch+CUDA":<16}{"OK" if torch_ok else "MISSING":<9}{"(torch with cuda)" if torch_ok else "(pip install torch --index-url https://download.pytorch.org/whl/cu124)"}')
    missing = [name for name, _, ok in rows if not ok]
    if not torch_ok:
        missing.append("PyTorch")
    print()
    if missing:
        print("Missing: " + ", ".join(missing))
        print("Order: install PyTorch -> compare -> annotate -> prepare -> train -> infer")
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
    p.add_argument("mode", choices=["check", "compare", "annotate", "prepare", "train", "infer"])
    p.add_argument("--interactive", action="store_true", help="pick files with dialogs instead of defaults")
    p.add_argument("--video", type=Path); p.add_argument("--roi", type=Path)
    p.add_argument("--labels", type=Path); p.add_argument("--dataset", type=Path)
    p.add_argument("--model", type=Path); p.add_argument("--infer-out", type=Path, dest="infer_out")
    p.add_argument("--compare-out", type=Path, dest="compare_out")
    p.add_argument("--arena-width-cm", type=float, dest="arena_width_cm")
    p.add_argument("--arena-height-cm", type=float, dest="arena_height_cm")
    p.add_argument("--max-labels", type=int, default=100)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()

    v = dict(DEFAULTS)
    for key in ("video", "roi", "labels", "dataset", "model", "infer_out", "compare_out", "arena_width_cm", "arena_height_cm"):
        value = getattr(a, key, None)
        if value is not None:
            v[key] = value
    if a.interactive:
        v = apply_dialogs(a.mode, v)

    if a.mode == "check":
        check(v); return
    script, argv = command(a.mode, v, a)
    print(f"$ python {script.relative_to(ROOT)} {' '.join(argv)}")
    raise SystemExit(subprocess.run([sys.executable, str(script), *argv]).returncode)


if __name__ == "__main__":
    main()