"""Compatibility helpers for labels created by older GUI revisions."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "excluded"}
    return bool(value)


def leaf_name(value) -> str:
    """Return a basename for Windows or POSIX paths on either operating system."""
    return str(value or "").strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def video_matches(value, target: str | Path) -> bool:
    """Match legacy full paths, basenames and time-only video identifiers."""
    candidate = leaf_name(value).casefold()
    wanted = leaf_name(target).casefold()
    if not candidate:
        return False
    if candidate == wanted:
        return True
    cstem = " ".join(candidate.rsplit(".", 1)[0].replace("_", " ").split())
    wstem = " ".join(wanted.rsplit(".", 1)[0].replace("_", " ").split())
    if cstem == wstem:
        return True
    # Early CSV revisions sometimes stored only the final timestamp token.
    return min(len(cstem), len(wstem)) >= 6 and (cstem.endswith(wstem) or wstem.endswith(cstem))


def video_mask(series: pd.Series, target: str | Path) -> pd.Series:
    return series.map(lambda value: video_matches(value, target))


def normalize_polygon(value) -> np.ndarray:
    """Accept Nx2, Nx1x2, flat coordinates and legacy {points: ...} JSON."""
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = value.get("points", value.get("polygon", value.get("polygon_px")))
    polygon = np.asarray(value, dtype=np.float32)
    if polygon.size < 6 or polygon.size % 2:
        raise ValueError("polygon needs at least three x/y points")
    polygon = polygon.reshape(-1, 2)
    if len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("polygon contains too few or non-finite points")
    return polygon


def atomic_upsert_polygon(path: str | Path, video: str | Path, frame: int,
                          polygon, exclude: bool = False,
                          source: str = "contact_sheet_edit") -> Path:
    """Upsert one polygon, verify the CSV, and preserve the previous file."""
    path = Path(path)
    columns = ["frame", "polygon_px", "exclude", "video", "source"]
    try:
        old = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=columns)
    except pd.errors.EmptyDataError:
        old = pd.DataFrame(columns=columns)
    if "video" not in old:
        old["video"] = Path(video).name
    if "frame" not in old:
        old["frame"] = np.nan
    matches = video_mask(old["video"], video) & (pd.to_numeric(old["frame"], errors="coerce") == int(frame))
    previous = old.loc[matches].iloc[-1].to_dict() if matches.any() else {}
    previous.update({"frame": int(frame),
                     "polygon_px": json.dumps(normalize_polygon(polygon).round(1).tolist()),
                     "exclude": bool(exclude), "video": Path(video).name,
                     "source": source})
    merged = pd.concat([old.loc[~matches], pd.DataFrame([previous])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    backup = path.with_suffix(path.suffix + ".bak")
    merged.to_csv(temporary, index=False)
    check = pd.read_csv(temporary)
    saved = video_mask(check["video"], video) & (pd.to_numeric(check["frame"], errors="coerce") == int(frame))
    if saved.sum() != 1:
        temporary.unlink(missing_ok=True)
        raise IOError(f"save verification failed for {Path(video).name} frame {frame}")
    normalize_polygon(check.loc[saved, "polygon_px"].iloc[0])
    if path.is_file():
        shutil.copy2(path, backup)
    temporary.replace(path)
    return backup


def atomic_upsert_head(path: str | Path, video: str | Path, frame: int,
                       timestamp_sec: float, head, reflection,
                       exclude: bool = False,
                       source: str = "head_result_correction",
                       head_verified: bool = False,
                       reflection_verified: bool = False) -> Path:
    """Atomically upsert one manual head/reflection pair across many videos."""
    path = Path(path)
    columns = ["frame", "timestamp_sec", "head_x_cm", "head_y_cm",
               "reflection_x_cm", "reflection_y_cm", "exclude",
               "reflection_present", "head_present", "head_verified",
               "reflection_verified", "video", "source"]
    try:
        old = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=columns)
    except pd.errors.EmptyDataError:
        old = pd.DataFrame(columns=columns)
    if "video" not in old:
        old["video"] = Path(video).name
    if "frame" not in old:
        old["frame"] = np.nan
    matches = video_mask(old.video, video) & (pd.to_numeric(old.frame, errors="coerce") == int(frame))
    previous = old.loc[matches].iloc[-1].to_dict() if matches.any() else {}
    h = np.asarray(head, float) if head is not None else np.asarray([np.nan, np.nan])
    r = np.asarray(reflection, float) if reflection is not None else np.asarray([np.nan, np.nan])
    previous.update({"frame": int(frame), "timestamp_sec": float(timestamp_sec),
                     "head_x_cm": h[0], "head_y_cm": h[1],
                     "reflection_x_cm": r[0], "reflection_y_cm": r[1],
                     "exclude": bool(exclude), "reflection_present": bool(np.isfinite(r).all()),
                     "head_present": bool(np.isfinite(h).all()),
                     "head_verified": bool(head_verified),
                     "reflection_verified": bool(reflection_verified),
                     "video": Path(video).name,
                     "source": source})
    merged = pd.concat([old.loc[~matches], pd.DataFrame([previous])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    backup = path.with_suffix(path.suffix + ".bak")
    merged.to_csv(temporary, index=False, float_format="%.5f")
    check = pd.read_csv(temporary)
    saved = video_mask(check.video, video) & (pd.to_numeric(check.frame, errors="coerce") == int(frame))
    if saved.sum() != 1:
        temporary.unlink(missing_ok=True)
        raise IOError(f"head save verification failed for {Path(video).name} frame {frame}")
    if path.is_file():
        shutil.copy2(path, backup)
    temporary.replace(path)
    return backup
