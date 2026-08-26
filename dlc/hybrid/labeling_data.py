"""Inspect and select extracted DLC image folders for napari labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class LabelFolder:
    path: Path
    image_count: int
    has_collected_data: bool


def inspect_label_folders(config_path: Path) -> list[LabelFolder]:
    data_dir = config_path.resolve().parent / "labeled-data"
    if not data_dir.is_dir():
        return []
    folders = []
    for folder in sorted((path for path in data_dir.iterdir() if path.is_dir()), key=lambda p: p.name.lower()):
        image_count = sum(
            1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        collected = any(folder.glob("CollectedData_*.h5")) or any(folder.glob("CollectedData_*.csv"))
        folders.append(LabelFolder(folder.resolve(), image_count, collected))
    return folders


def choose_label_folder(config_path: Path) -> tuple[LabelFolder, list[LabelFolder]]:
    folders = inspect_label_folders(config_path)
    valid = [folder for folder in folders if folder.image_count > 0]
    if not valid:
        details = ", ".join(f"{folder.path.name}={folder.image_count}" for folder in folders) or "no folders"
        raise RuntimeError(
            "No extracted images are available for labeling under the DLC project's labeled-data directory "
            f"({details}). Run 'Extract DLC frames' again and inspect the extraction report."
        )
    pending = [folder for folder in valid if not folder.has_collected_data]
    return (pending[0] if pending else valid[0]), valid


def extraction_summary(config_path: Path) -> str:
    folders = inspect_label_folders(config_path)
    total = sum(folder.image_count for folder in folders)
    if total == 0:
        raise RuntimeError(
            "DeepLabCut returned from frame extraction but produced zero readable images. "
            "Check that the project's video paths still exist and that the videos decode correctly."
        )
    details = ", ".join(f"{folder.path.name}: {folder.image_count}" for folder in folders)
    return f"Extracted-frame verification passed: {total} images across {len(folders)} folders ({details})"
