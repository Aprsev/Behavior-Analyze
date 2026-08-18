"""Compatibility helpers for labels created by older GUI revisions."""
from __future__ import annotations

import json
from pathlib import Path

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
