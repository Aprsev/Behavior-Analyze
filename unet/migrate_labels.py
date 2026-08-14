#!/usr/bin/env python3
"""Recover the video column for an old manual_torso_constraints.csv.

Labels saved before the multi-video fix have no video column, so equal frame
numbers from different recordings silently overwrote each other. This tool
assigns each label row to the video whose screening CSV contains that frame.

A frame number that appears in the screening rows of more than one video
cannot be assigned reliably; such rows are reported as conflicts and should
be re-labelled (run annotate on that video again after migration).

Usage:
    python unet/migrate_labels.py --labels traditional/results/manual_torso_constraints.csv --screening traditional/results/screening.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, help="old manual_torso_constraints.csv")
    p.add_argument("--screening", required=True, help="screening.csv with video, frame, exclude")
    a = p.parse_args()

    labels = pd.read_csv(a.labels)
    if "video" in labels.columns:
        print("Labels already have a video column; nothing to migrate.")
        return

    screening = pd.read_csv(a.screening)
    owner: dict[int, set[str]] = {}
    for _, row in screening.iterrows():
        owner.setdefault(int(row.frame), set()).add(str(row.video))

    assigned, conflicts = [], []
    for _, row in labels.iterrows():
        videos = owner.get(int(row.frame), set())
        if len(videos) == 1:
            row = row.copy(); row["video"] = next(iter(videos)); assigned.append(row)
        else:
            row = row.copy(); row["video"] = ""; conflicts.append(row)

    out = pd.DataFrame(assigned)
    out.to_csv(a.labels, index=False)
    print(f"Assigned {len(assigned)} rows to their video.")
    if conflicts:
        print(f"CONFLICT: {len(conflicts)} rows have a frame number present in "
              f"multiple videos and cannot be assigned. Re-label them:")
        for _, row in pd.DataFrame(conflicts).iterrows():
            print(f"  frame {int(row.frame)}")
    else:
        print("No conflicts - all rows assigned cleanly.")


if __name__ == "__main__":
    main()