"""Dataset import, low-confidence mining, and YOLO checkpoint fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from dlc.hybrid._hybrid_pipeline_impl import BOX_COLUMNS, load_box_labels, save_box_labels


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Candidate:
    video: str
    frame: int
    fps: float
    confidence: float
    box: tuple[float, float, float, float] | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=[*BOX_COLUMNS, "model_confidence", "review_batch", "imported_split"])


def _current_labels(root: Path) -> pd.DataFrame:
    path = root / "box_labels.csv"
    return load_box_labels(path) if path.is_file() else _empty_labels()


def _known_hashes(rows: pd.DataFrame) -> set[str]:
    hashes: set[str] = set()
    if "image_sha256" in rows:
        hashes.update(str(value) for value in rows.image_sha256.dropna() if str(value))
    for value in rows.get("image", pd.Series(dtype=str)).dropna():
        path = Path(str(value))
        if path.is_file():
            hashes.add(_sha256(path))
    return hashes


def _copy_image(source: Path, destination_root: Path, digest: str) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"imported_{digest[:16]}{source.suffix.lower()}"
    if not destination.is_file():
        shutil.copy2(source, destination)
    return destination.resolve()


def _native_rows(source_root: Path) -> pd.DataFrame | None:
    labels = source_root / "box_labels.csv"
    if not labels.is_file():
        return None
    rows = load_box_labels(labels)
    return rows.loc[
        ~rows.exclude & rows[["x1", "y1", "x2", "y2"]].notna().all(axis=1)
    ].copy()


def _standard_yolo_rows(source_root: Path) -> pd.DataFrame:
    images_root, labels_root = source_root / "images", source_root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(
            f"Expected either box_labels.csv or standard images/ and labels/ folders in {source_root}"
        )
    rows: list[dict[str, Any]] = []
    for image in sorted(path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        relative = image.relative_to(images_root)
        label = (labels_root / relative).with_suffix(".txt")
        if not label.is_file():
            continue
        decoded = cv2.imread(str(image))
        if decoded is None:
            continue
        height, width = decoded.shape[:2]
        boxes: list[tuple[float, float, float, float]] = []
        for line in label.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 5 or int(float(values[0])) != 0:
                continue
            xc, yc, bw, bh = map(float, values[1:5])
            boxes.append((
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            ))
        if not boxes:
            continue
        box = max(boxes, key=lambda item: max(0, item[2] - item[0]) * max(0, item[3] - item[1]))
        split = relative.parts[0] if len(relative.parts) > 1 else "imported"
        rows.append({
            "video": f"imported:{source_root.name}:{split}:{image.stem}",
            "frame": 0,
            "image": str(image.resolve()),
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            "source": "imported_standard_yolo", "confidence": 1.0,
            "exclude": False, "reviewed": True, "imported_split": split,
        })
    return pd.DataFrame(rows)


def import_labeled_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge a native or standard YOLO dataset into the current audit CSV."""
    active = cfg.get("active_learning", {})
    source_root = Path(active.get("import_dataset_dir", "")).expanduser().resolve()
    target_root = Path(cfg["yolo"]["dataset_dir"]).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Existing labeled dataset directory is missing: {source_root}")
    if source_root == target_root:
        rows = _current_labels(target_root)
        return {"imported": 0, "duplicates": len(rows), "total": len(rows), "mode": "already_current"}

    incoming = _native_rows(source_root)
    mode = "native_box_labels"
    if incoming is None:
        incoming = _standard_yolo_rows(source_root)
        mode = "standard_yolo"
    current = _current_labels(target_root)
    known = _known_hashes(current)
    destination_images = target_root / "source_images"
    imported: list[dict[str, Any]] = []
    duplicates = 0
    for _, row in incoming.iterrows():
        source = Path(str(row.image))
        if not source.is_absolute():
            source = (source_root / source).resolve()
        if not source.is_file():
            print(f"Skipping missing imported image: {source}", flush=True)
            continue
        digest = _sha256(source)
        if digest in known:
            duplicates += 1
            continue
        known.add(digest)
        values = row.to_dict()
        values.update({
            "image": str(_copy_image(source, destination_images, digest)),
            "image_sha256": digest,
            "source": "imported_dataset" if mode == "native_box_labels" else values.get("source"),
            "confidence": float(values.get("confidence", 1.0)),
            "model_confidence": values.get("model_confidence", np.nan),
            "exclude": False,
            "reviewed": True,
            "review_batch": "",
        })
        imported.append(values)
    if imported:
        combined = pd.concat([current, pd.DataFrame(imported)], ignore_index=True, sort=False)
    else:
        combined = current
    target_root.mkdir(parents=True, exist_ok=True)
    save_box_labels(target_root / "box_labels.csv", combined)
    report = {
        "source": str(source_root), "target": str(target_root), "mode": mode,
        "imported": len(imported), "duplicates": duplicates, "total": len(combined),
    }
    (target_root / "last_import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Imported labeled dataset: {report}", flush=True)
    return report


def select_low_confidence(
    candidates: Iterable[Candidate], count: int, minimum_gap_sec: float,
) -> list[Candidate]:
    """Select lowest confidence candidates while suppressing near-duplicate times."""
    ranked = sorted(candidates, key=lambda item: (item.confidence, item.video, item.frame))
    selected: list[Candidate] = []
    for item in ranked:
        gap = max(0, int(round(minimum_gap_sec * item.fps)))
        if any(other.video == item.video and abs(other.frame - item.frame) < gap for other in selected):
            continue
        selected.append(item)
        if len(selected) >= count:
            return selected
    for item in ranked:
        if item not in selected:
            selected.append(item)
        if len(selected) >= count:
            break
    return selected


def _predict_candidates(model, video: Path, cfg: dict[str, Any]) -> list[Candidate]:
    active, yolo = cfg.get("active_learning", {}), cfg.get("yolo", {})
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open new training video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    candidate_count = min(total, int(active.get("candidate_frames_per_video", 300)))
    indices = np.unique(np.linspace(0, total - 1, candidate_count, dtype=int))
    results: list[Candidate] = []
    for position, frame_index in enumerate(indices, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        prediction = model.predict(
            frame,
            conf=float(active.get("min_prediction_confidence", 0.001)),
            iou=float(yolo.get("iou", 0.7)),
            imgsz=int(yolo.get("image_size", 640)),
            device=yolo.get("device", "auto"),
            verbose=False,
        )[0]
        box = None
        confidence = 0.0
        if prediction.boxes is not None and len(prediction.boxes):
            xyxy = prediction.boxes.xyxy.detach().cpu().numpy()
            confs = prediction.boxes.conf.detach().cpu().numpy()
            classes = prediction.boxes.cls.detach().cpu().numpy().astype(int)
            mouse = np.flatnonzero(classes == 0)
            if len(mouse):
                chosen = mouse[int(np.argmax(confs[mouse]))]
                box = tuple(map(float, xyxy[chosen]))
                confidence = float(confs[chosen])
        results.append(Candidate(str(video.resolve()), int(frame_index), fps, confidence, box))
        if position % 50 == 0:
            print(f"Scored {position}/{len(indices)} candidates from {video.name}", flush=True)
    cap.release()
    return results


def _read_frame(video: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video}")
    return frame


def mine_low_confidence_frames(cfg: dict[str, Any]) -> dict[str, Any]:
    """Use the current checkpoint to create a manually reviewable active-learning batch."""
    from ultralytics import YOLO

    active, yolo = cfg.get("active_learning", {}), cfg.get("yolo", {})
    videos = [Path(value).expanduser().resolve() for value in active.get("new_videos", [])]
    if not videos:
        raise ValueError("No new active-learning videos configured")
    missing = [str(video) for video in videos if not video.is_file()]
    if missing:
        raise FileNotFoundError("Missing new videos: " + ", ".join(missing))
    checkpoint = Path(yolo.get("trained_model", "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Select the existing YOLO best.pt first: {checkpoint}")
    dataset = Path(yolo["dataset_dir"]).expanduser().resolve()
    rows = _current_labels(dataset)
    existing = {
        (str(Path(str(row.video)).resolve()), int(row.frame))
        for _, row in rows.iterrows() if str(row.get("video", ""))
    }
    model = YOLO(str(checkpoint))
    candidates: list[Candidate] = []
    for video in videos:
        candidates.extend(
            item for item in _predict_candidates(model, video, cfg)
            if (item.video, item.frame) not in existing
        )
    requested = int(active.get("frames_to_review", 80))
    selected = select_low_confidence(candidates, requested, float(active.get("minimum_gap_sec", 0.5)))
    if not selected:
        raise RuntimeError("No new candidate frames remain after duplicate filtering")
    batch_id = datetime.now().strftime("active_%Y%m%d_%H%M%S")
    image_root = dataset / "source_images"
    new_rows = []
    for item in selected:
        frame = _read_frame(Path(item.video), item.frame)
        token = hashlib.sha256(item.video.encode()).hexdigest()[:10]
        image = image_root / f"{batch_id}_{token}_{item.frame:08d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        box = item.box or (np.nan, np.nan, np.nan, np.nan)
        new_rows.append({
            "video": item.video, "frame": item.frame, "image": str(image.resolve()),
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            "source": "active_yolo" if item.box else "active_yolo_missing",
            "confidence": item.confidence, "model_confidence": item.confidence,
            "exclude": False, "reviewed": False, "review_batch": batch_id,
            "dataset_split": "train_source", "image_sha256": _sha256(image),
        })
    combined = pd.concat([rows, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    save_box_labels(dataset / "box_labels.csv", combined)
    report = {
        "batch_id": batch_id, "checkpoint": str(checkpoint),
        "videos": [str(video) for video in videos], "candidates_scored": len(candidates),
        "selected": len(new_rows),
        "confidence_min": min(item.confidence for item in selected),
        "confidence_max": max(item.confidence for item in selected),
        "missing_detections": sum(item.box is None for item in selected),
    }
    reports = dataset / "active_learning_batches"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{batch_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Active-learning batch: {report}", flush=True)
    return report


def latest_review_batch(labels_csv: Path) -> str:
    rows = load_box_labels(labels_csv)
    if "review_batch" not in rows:
        raise ValueError("No active-learning review batch exists")
    values = [str(value) for value in rows.review_batch.dropna() if str(value).startswith("active_")]
    if not values:
        raise ValueError("No active-learning review batch exists")
    return max(values)
