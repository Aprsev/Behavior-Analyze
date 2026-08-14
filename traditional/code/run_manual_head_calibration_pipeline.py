#!/usr/bin/env python3
"""Run automatic candidates -> manual calibration -> lightweight training -> final CSV.

The manual labelling window is intentionally the middle stage. It will not
continue until you press Q/Esc, which saves the labels. Re-running the command
keeps existing labels and lets you revise/add more examples.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def command(script: str, *arguments: str) -> list[str]:
    return [sys.executable, str(Path(__file__).with_name(script)), *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--roi-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arena-width-cm", type=float, required=True)
    parser.add_argument("--arena-height-cm", type=float, required=True)
    parser.add_argument("--labels", default="", help="Existing or new label CSV; defaults inside output directory")
    parser.add_argument("--max-labels", type=int, default=100)
    parser.add_argument("--torso-labels", default="", help="Optional manual torso-box CSV used to exclude fibre on labelled frames")
    parser.add_argument("--skip-auto", action="store_true", help="Reuse existing automatic comparison CSV")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    comparison = out / "head_method_comparison.csv"
    labels = Path(args.labels) if args.labels else out / "manual_head_anchor_labels.csv"
    model, metrics, final = out / "head_calibrator.joblib", out / "head_calibrator_metrics.json", out / "trajectory_manual_calibrated.csv"
    common = ["--input", args.input, "--roi-json", args.roi_json, "--arena-width-cm", str(args.arena_width_cm), "--arena-height-cm", str(args.arena_height_cm)]
    if not args.skip_auto:
        auto_args = [*common, "--output-dir", str(out)]
        if args.torso_labels:
            auto_args.extend(["--torso-labels", args.torso_labels])
        subprocess.run(command("compare_head_methods.py", *auto_args), check=True)
    if not comparison.exists():
        raise FileNotFoundError(f"Automatic comparison CSV missing: {comparison}")
    subprocess.run(command("annotate_head_anchors.py", *common, "--comparison-csv", str(comparison), "--output", str(labels), "--max-labels", str(args.max_labels)), check=True)
    subprocess.run(command("train_head_calibrator.py", "--comparison-csv", str(comparison), "--labels", str(labels), "--model-output", str(model), "--metrics-output", str(metrics)), check=True)
    subprocess.run(command("finalize_head_trajectory.py", "--comparison-csv", str(comparison), "--model", str(model), "--output", str(final), "--labels", str(labels), "--input", args.input, "--roi-json", args.roi_json, "--arena-width-cm", str(args.arena_width_cm), "--arena-height-cm", str(args.arena_height_cm), "--output-video", str(out / "annotated_manual_calibrated.mp4")), check=True)
    print(f"\nCOMPLETE\nFinal trajectory: {final}\nModel metrics: {metrics}\nLabels: {labels}")


if __name__ == "__main__":
    main()
