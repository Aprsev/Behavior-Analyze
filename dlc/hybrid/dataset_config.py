"""Hybrid configuration migration and explicit train/test video sets."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _resolve(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (base / path).resolve())


def _resolve_optional_model(value: str, base: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return _resolve(value, base)
    return value


def pair_video_rois(videos: Iterable[str], rois: Iterable[str], split: str) -> list[dict[str, str]]:
    video_list = list(videos)
    roi_list = list(rois)
    if not video_list:
        return []
    if len(roi_list) == 1:
        roi_list *= len(video_list)
    if len(roi_list) != len(video_list):
        raise ValueError(
            f"{split}_rois must contain either one shared ROI or exactly one ROI per "
            f"video ({len(video_list)} videos, {len(roi_list)} ROI files)"
        )
    return [
        {"split": split, "video": video, "roi_json": roi}
        for video, roi in zip(video_list, roi_list)
    ]


def load_hybrid_config(path: Path) -> dict[str, Any]:
    """Load the current schema and migrate configurations saved before modularization."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    config_dir = path.resolve().parent
    old_schema = "training_videos" not in raw and "test_videos" not in raw
    # Old configs lived one directory higher. After the package move their ../
    # paths must still be interpreted relative to dlc/, not dlc/hybrid/.
    relative_base = config_dir.parent if old_schema and config_dir.name == "hybrid" else config_dir

    training_raw = _items(raw.get("training_videos")) or _items(raw.get("videos"))
    if not training_raw and raw.get("video"):
        training_raw = [str(raw["video"])]
    test_raw = _items(raw.get("test_videos"))
    training_rois_raw = _items(raw.get("training_rois")) or _items(raw.get("roi_json"))
    test_rois_raw = _items(raw.get("test_rois"))
    if test_raw and not test_rois_raw:
        test_rois_raw = training_rois_raw[:1]

    training = [_resolve(value, relative_base) for value in training_raw]
    testing = [_resolve(value, relative_base) for value in test_raw]
    training_rois = [_resolve(value, relative_base) for value in training_rois_raw]
    test_rois = [_resolve(value, relative_base) for value in test_rois_raw]

    cfg = deepcopy(raw)
    cfg["training_videos"] = training
    cfg["test_videos"] = testing
    cfg["training_rois"] = training_rois
    cfg["test_rois"] = test_rois
    cfg["video_sets"] = {
        "train": pair_video_rois(training, training_rois, "train"),
        "test": pair_video_rois(testing, test_rois, "test"),
    }
    # The supervised DLC branch consumes only training videos.
    cfg["videos"] = training
    cfg["video"] = training[0] if training else ""
    cfg["roi_json"] = training_rois[0] if training_rois else ""

    for key in ("output_dir", "project_config", "working_directory"):
        if cfg.get(key):
            cfg[key] = _resolve(str(cfg[key]), relative_base)
    yolo = cfg.setdefault("yolo", {})
    if yolo.get("dataset_dir"):
        yolo["dataset_dir"] = _resolve(str(yolo["dataset_dir"]), relative_base)
    yolo["base_model"] = _resolve_optional_model(str(yolo.get("base_model", "yolo26n.pt")), relative_base)
    if yolo.get("trained_model"):
        yolo["trained_model"] = _resolve(str(yolo["trained_model"]), relative_base)
    sr = cfg.setdefault("super_resolution", {})
    if sr.get("model_path"):
        sr["model_path"] = _resolve(str(sr["model_path"]), relative_base)
    return cfg


def records(cfg: dict[str, Any], split: str) -> list[dict[str, str]]:
    value = cfg.get("video_sets", {}).get(split, [])
    return [dict(item) for item in value]


def evaluation_records(cfg: dict[str, Any]) -> list[dict[str, str]]:
    return records(cfg, "test") or records(cfg, "train")


def config_for_record(cfg: dict[str, Any], record: dict[str, str]) -> dict[str, Any]:
    result = deepcopy(cfg)
    result["videos"] = [record["video"]]
    result["video"] = record["video"]
    result["roi_json"] = record["roi_json"]
    return result


def validate_files(cfg: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if not records(cfg, "train"):
        problems.append("No training videos configured")
    for split in ("train", "test"):
        for index, item in enumerate(records(cfg, split), 1):
            if not Path(item["video"]).is_file():
                problems.append(f"{split} video {index} is missing: {item['video']}")
            if not Path(item["roi_json"]).is_file():
                problems.append(f"{split} ROI {index} is missing: {item['roi_json']}")
    return problems
