"""Synchronize the Hybrid training-video set into an existing DLC project."""

from __future__ import annotations

from pathlib import Path


def _existing_settings(video: Path, mappings: list[dict]) -> dict | None:
    matches = []
    for mapping in mappings:
        for raw, settings in mapping.items():
            if Path(str(raw)).stem == video.stem and isinstance(settings, dict):
                matches.append(dict(settings))
    return matches[0] if matches else None


def sync_training_videos(hybrid_cfg: dict, project_config: Path) -> list[Path]:
    """Write verified source videos into DLC video_sets using stable absolute paths."""
    import cv2
    from deeplabcut.utils import auxiliaryfunctions

    videos = [Path(value).expanduser().resolve() for value in hybrid_cfg.get("videos", [])]
    if not videos:
        raise ValueError("No training videos are configured for the supervised DLC project")
    missing = [path for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DLC training videos: " + ", ".join(map(str, missing)))
    stems = [path.stem.casefold() for path in videos]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "DLC labeled-data folders are named from video stems; training videos must have unique filenames"
        )

    project = auxiliaryfunctions.read_config(str(project_config))
    existing = [
        value for value in (project.get("video_sets", {}), project.get("video_sets_original", {}))
        if isinstance(value, dict)
    ]
    synchronized = {}
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _frame = capture.read()
        capture.release()
        if not ok or width <= 0 or height <= 0:
            raise RuntimeError(f"DLC training video cannot be decoded: {video}")
        settings = _existing_settings(video, existing) or {"crop": f"0, {width}, 0, {height}"}
        synchronized[str(video)] = settings

    project["video_sets"] = synchronized
    if "video_sets_original" in project:
        project["video_sets_original"] = synchronized.copy()
    auxiliaryfunctions.write_config(str(project_config), project)
    print(
        f"Synchronized {len(videos)} verified training videos into DLC project video_sets:",
        flush=True,
    )
    for video in videos:
        print(f"- {video}", flush=True)
    return videos
