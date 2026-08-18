#!/usr/bin/env python3
"""Train the fibre-aware multi-task U-Net from an authoritative manifest.

New models see two channels: raw grayscale keeps a stationary mouse visible;
background residual supplies motion contrast. Existing anatomical Head and
Reflection labels supervise separate heatmap decoders. Training may warm-start
from any compatible checkpoint but always writes a new timestamped .pt file.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import UNet, load_compatible_weights, unpack_outputs
from postprocess import model_input

BG_GAIN = 2.0


def _stem_of(name: str) -> str:
    return name[:-12]


def manifest_files(root: Path) -> list[str]:
    """Use dataset.json as truth; never train on orphan PNG files."""
    path = root / "dataset.json"
    if not path.is_file():
        raise ValueError(f"Missing {path}; rerun prepare")
    data = json.loads(path.read_text(encoding="utf-8"))
    files: list[str] = []
    for video in data.get("videos", []):
        samples = video.get("samples")
        if not samples:
            stem = video.get("stem") or Path(video["video"]).stem.replace(" ", "_")
            samples = [f"{stem}_{int(f):07d}.png" for f in video.get("frames", [])]
        excluded = {int(frame) for frame in video.get("excluded_frames", [])}
        leaked = [name for name in samples
                  if int(Path(str(name)).stem.rsplit("_", 1)[-1]) in excluded]
        if leaked:
            raise ValueError(f"Manifest contains {len(leaked)} excluded training samples; "
                             f"rerun rebuild. First: {leaked[0]}")
        files.extend(str(name) for name in samples)
    files = sorted(dict.fromkeys(files))
    missing = [f for f in files if not (root / "images" / f).is_file()
               or not (root / "masks" / f).is_file()]
    if missing:
        raise ValueError(f"Manifest references {len(missing)} missing pairs; rerun prepare. First: {missing[0]}")
    return files


def load_backgrounds(root: Path, files: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in files:
        stem = _stem_of(name)
        if stem not in result:
            bg = cv2.imread(str(root / "backgrounds" / f"{stem}.png"), 0)
            if bg is not None:
                result[stem] = bg
    return result


def has_head_heatmap(path: Path) -> bool:
    """Optional head targets may be absent in legacy torso-only datasets."""
    if not path.is_file():
        return False
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image is not None and bool(image.max() > 0)


def temporal_split(files: list[str], fraction: float = 0.2) -> tuple[list[str], list[str]]:
    """Hold out one contiguous time block per video instead of adjacent leakage."""
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(_stem_of(f), []).append(f)
    train: list[str] = []; val: list[str] = []
    for _, rows in sorted(groups.items()):
        rows = sorted(rows)
        n = max(1, round(len(rows) * fraction))
        start = max(0, (len(rows) - n) // 2)
        val.extend(rows[start:start + n])
        train.extend(rows[:start] + rows[start + n:])
    return train, val


def ensure_keypoint_coverage(root: Path, train: list[str], val: list[str], folder: str) -> None:
    """Keep at least one labelled point in both partitions when possible."""
    labelled = [name for name in train + val if has_head_heatmap(root / folder / name)]
    if len(labelled) < 2:
        return
    train_labelled = [name for name in train if name in labelled]
    val_labelled = [name for name in val if name in labelled]
    if not train_labelled and len(val_labelled) > 1:
        name = val_labelled[0]; val.remove(name); train.append(name)
    elif not val_labelled and len(train_labelled) > 1:
        name = train_labelled[-1]; train.remove(name); val.append(name)


class Pairs(Dataset):
    def __init__(self, root: Path, files: list[str], augment: bool,
                 backgrounds: dict[str, np.ndarray], in_channels: int):
        self.root, self.files, self.augment = root, files, augment
        self.backgrounds, self.in_channels = backgrounds, in_channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name = self.files[index]
        raw = cv2.imread(str(self.root / "images" / name), 0)
        mask = cv2.imread(str(self.root / "masks" / name), 0)
        head_path = self.root / "heads" / name
        head = cv2.imread(str(head_path), 0) if head_path.is_file() else None
        if head is None:
            head = np.zeros_like(mask)
        reflection_path = self.root / "reflections" / name
        reflection = cv2.imread(str(reflection_path), 0) if reflection_path.is_file() else None
        if reflection is None:
            reflection = np.zeros_like(mask)
        bg = self.backgrounds.get(_stem_of(name))
        if bg is not None and bg.shape != raw.shape:
            bg = cv2.resize(bg, raw.shape[::-1], interpolation=cv2.INTER_AREA)
        x = model_input(raw, bg, self.in_channels, BG_GAIN).transpose(1, 2, 0)
        if x.ndim == 2:
            x = x[..., None]

        if self.augment:
            k = random.randint(0, 3)
            if k:
                x, mask, head, reflection = (np.rot90(x, k).copy(), np.rot90(mask, k).copy(),
                                              np.rot90(head, k).copy(), np.rot90(reflection, k).copy())
            if random.random() < .5:
                x, mask, head, reflection = (np.fliplr(x).copy(), np.fliplr(mask).copy(),
                                              np.fliplr(head).copy(), np.fliplr(reflection).copy())
            if random.random() < .5:
                x, mask, head, reflection = (np.flipud(x).copy(), np.flipud(mask).copy(),
                                              np.flipud(head).copy(), np.flipud(reflection).copy())
            angle = random.uniform(-12, 12)
            matrix = cv2.getRotationMatrix2D((x.shape[1] / 2, x.shape[0] / 2), angle, 1)
            channels = [cv2.warpAffine(x[..., c], matrix, x.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
                        for c in range(x.shape[2])]
            x = np.stack(channels, axis=2)
            mask = cv2.warpAffine(mask, matrix, mask.shape[::-1], flags=cv2.INTER_NEAREST)
            head = cv2.warpAffine(head, matrix, head.shape[::-1], flags=cv2.INTER_LINEAR)
            reflection = cv2.warpAffine(reflection, matrix, reflection.shape[::-1], flags=cv2.INTER_LINEAR)
            gain = random.uniform(.88, 1.12); offset = random.uniform(-6, 6)
            x[..., 0] = np.clip(x[..., 0].astype(np.float32) * gain + offset, 0, 255)

        x_t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0
        mask_t = torch.from_numpy((mask[None] > 127).copy()).float()
        head_t = torch.from_numpy((head[None].astype(np.float32) / 255.0).copy())
        head_valid = torch.tensor(float(head.max() > 0), dtype=torch.float32)
        reflection_t = torch.from_numpy((reflection[None].astype(np.float32) / 255.0).copy())
        reflection_valid = torch.tensor(float(reflection.max() > 0), dtype=torch.float32)
        return x_t, mask_t, head_t, head_valid, reflection_t, reflection_valid


def soft_dice(logits, target):
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    return ((2 * (prob * target).sum(dims) + 1) /
            ((prob + target).sum(dims) + 1)).mean()


def hard_negative_loss(logits, target):
    """Focus learning on fibre-like false positives instead of easy arena pixels."""
    probability = torch.sigmoid(logits)
    negative_loss = -torch.log((1.0 - probability).clamp_min(1e-6)) * (1.0 - target)
    flat = negative_loss.flatten(1)
    positive_pixels = int(target.sum().detach().item() / max(target.shape[0], 1))
    k = min(flat.shape[1], max(256, positive_pixels * 2))
    return flat.topk(k, dim=1).values.mean()


def _keypoint_errors(logits, target, valid) -> list[float]:
    if logits is None or not valid.any():
        return []
    pred = torch.sigmoid(logits).cpu().flatten(1).argmax(1)
    true = target.flatten(1).argmax(1)
    width = target.shape[-1]
    distance = torch.sqrt(((pred % width) - (true % width)).float() ** 2 +
                          ((pred // width) - (true // width)).float() ** 2)
    return distance[valid > 0].tolist()


def point_heatmap_loss(logits, target, valid):
    """Location-first loss that cannot win by predicting an all-zero map."""
    probability = torch.sigmoid(logits)
    shape_loss = (((probability - target) ** 2) * (1.0 + 32.0 * target)).mean((1, 2, 3))
    flat_target = target.flatten(1)
    distribution = flat_target / flat_target.sum(1, keepdim=True).clamp_min(1e-6)
    log_probability = F.log_softmax(logits.flatten(1), dim=1)
    localization = -(distribution * log_probability).sum(1) / np.log(logits[0].numel())
    per_sample = localization + 0.25 * shape_loss
    return (per_sample * valid).sum() / valid.sum().clamp_min(1)


def validation_score(dice: float, head_error, reflection_error, size: int) -> float:
    """Prefer accurate masks while breaking close Dice ties with keypoints."""
    penalty = 0.0
    if head_error is not None:
        penalty += 0.15 * float(head_error) / size
    if reflection_error is not None:
        penalty += 0.25 * float(reflection_error) / size
    return float(dice - penalty)


def evaluate(model, loader, device):
    model.eval(); dice_scores = []; head_errors = []; reflection_errors = []
    with torch.no_grad():
        for x, mask, head, valid, reflection, reflection_valid in loader:
            output = model(x.to(device))
            mask_logits, head_logits, reflection_logits = unpack_outputs(output)
            dice_scores.append(float(soft_dice(mask_logits, mask.to(device)).item()))
            head_errors.extend(_keypoint_errors(head_logits, head, valid))
            reflection_errors.extend(_keypoint_errors(reflection_logits, reflection, reflection_valid))
    return (float(np.mean(dice_scores)),
            float(np.mean(head_errors)) if head_errors else None,
            float(np.mean(reflection_errors)) if reflection_errors else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true", help="do not warm-start from --base-model")
    p.add_argument("--base-model", default="", help="optional old/new checkpoint used only for warm start")
    p.add_argument("--patience", type=int, default=18)
    a = p.parse_args()
    requested_lr = float(a.lr)
    a.lr = min(max(requested_lr, 1e-6), 3e-3)
    if a.lr != requested_lr:
        print(f"WARNING: requested learning rate {requested_lr:g} is unsafe for keypoint "
              f"training; clamped to {a.lr:g}")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    root = Path(a.dataset); files = manifest_files(root)
    if len(files) < 20:
        raise ValueError(f"Need >=20 masks, found {len(files)}")
    backgrounds = load_backgrounds(root, files)
    stems = {_stem_of(f) for f in files}
    if len(backgrounds) != len(stems):
        raise ValueError(f"Only {len(backgrounds)}/{len(stems)} video backgrounds exist. Rerun prepare; "
                         "mixed raw/residual training is forbidden.")
    in_channels = 2
    sample_size = int(cv2.imread(str(root / "images" / files[0]), 0).shape[0])
    head_output = any(has_head_heatmap(root / "heads" / f) for f in files)
    reflection_output = any(has_head_heatmap(root / "reflections" / f) for f in files)
    train_files, val_files = temporal_split(files)
    ensure_keypoint_coverage(root, train_files, val_files, "heads")
    ensure_keypoint_coverage(root, train_files, val_files, "reflections")
    if not train_files or not val_files:
        raise ValueError("Training/validation split is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=in_channels, head_output=head_output,
                 reflection_output=reflection_output).to(device)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate_path = out / f"candidate_unet_reflection_{stamp}.pt"
    promoted_path = out / f"best_unet_reflection_{stamp}.pt"
    if candidate_path.exists() or promoted_path.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint")
    resumed = False
    base_path = Path(a.base_model) if a.base_model else out / "best_unet.pt"
    old = None
    base_had_head = base_had_reflection = False
    if base_path.is_file() and not a.fresh:
        old = torch.load(base_path, map_location="cpu")
        base_had_head = bool(old.get("head_output", False))
        base_had_reflection = bool(old.get("reflection_output", False))
        source_channel = 1 if bool(old.get("bg_subtract")) else 0
        loaded = load_compatible_weights(model, old["state_dict"], source_channel=source_channel)
        resumed = bool(loaded)
        print(f"Warm-started {len(loaded)} tensors from {base_path.name}; source checkpoint remains unchanged")

    kw = dict(num_workers=2, pin_memory=device == "cuda")
    train_loader = DataLoader(Pairs(root, train_files, True, backgrounds, in_channels),
                              batch_size=a.batch_size, shuffle=True, **kw)
    val_loader = DataLoader(Pairs(root, val_files, False, backgrounds, in_channels),
                            batch_size=a.batch_size, **kw)
    baseline_dice, baseline_head_error, baseline_reflection_error = \
        evaluate(model, val_loader, device) if resumed else (-1.0, None, None)
    baseline_score = validation_score(baseline_dice, baseline_head_error,
                                      baseline_reflection_error, sample_size) if resumed else -1.0
    if resumed:
        print(f"Current checkpoint on this validation split: Dice {baseline_dice:.5f}; "
              f"head {baseline_head_error}; reflection {baseline_reflection_error}")
    if resumed:
        new_point_parameters = []
        if head_output and not base_had_head and model.head_out is not None:
            new_point_parameters.extend(model.head_out.parameters())
        if reflection_output and not base_had_reflection and model.reflection_out is not None:
            new_point_parameters.extend(model.reflection_out.parameters())
        new_ids = {id(parameter) for parameter in new_point_parameters}
        inherited_parameters = [parameter for parameter in model.parameters()
                                if id(parameter) not in new_ids]
        groups = [{"params": inherited_parameters, "lr": min(a.lr, 2e-4)}]
        if new_point_parameters:
            groups.append({"params": new_point_parameters, "lr": a.lr})
            print(f"Using LR {a.lr:g} for {len(new_point_parameters)} new point-layer tensors; "
                  f"inherited layers use {min(a.lr, 2e-4):g}")
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    best = -1.0; best_score = -1e9; best_epoch = 0; best_head_error = None
    best_reflection_error = None; history = []
    for epoch in range(1, a.epochs + 1):
        model.train(); losses = []
        for x, mask, head, valid, reflection, reflection_valid in train_loader:
            x, mask = x.to(device), mask.to(device)
            head, valid = head.to(device), valid.to(device)
            reflection, reflection_valid = reflection.to(device), reflection_valid.to(device)
            output = model(x)
            mask_logits, head_logits, reflection_logits = unpack_outputs(output)
            loss = (bce(mask_logits, mask) + (1.0 - soft_dice(mask_logits, mask)) +
                    0.25 * hard_negative_loss(mask_logits, mask))
            if head_logits is not None and valid.any():
                loss = loss + 0.45 * point_heatmap_loss(head_logits, head, valid)
            if reflection_logits is not None and reflection_valid.any():
                loss = loss + 0.40 * point_heatmap_loss(
                    reflection_logits, reflection, reflection_valid)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        val_dice, head_error, reflection_error = evaluate(model, val_loader, device)
        score = validation_score(val_dice, head_error, reflection_error, sample_size)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": val_dice,
               "head_error_px": head_error, "reflection_error_px": reflection_error,
               "selection_score": score}
        history.append(row); print(json.dumps(row), flush=True)
        if score > best_score:
            best, best_score, best_epoch = val_dice, score, epoch
            best_head_error, best_reflection_error = head_error, reflection_error
            torch.save({"state_dict": model.state_dict(), "size": sample_size,
                        "val_dice": best, "head_error_px": head_error,
                        "reflection_error_px": reflection_error,
                        "selection_score": score, "in_channels": in_channels,
                        "dual_channel": True, "bg_subtract": True, "bg_gain": BG_GAIN,
                        "head_output": head_output, "reflection_output": reflection_output,
                        "checkpoint_version": 3,
                        "validation": "contiguous_temporal_block"}, candidate_path)
        if epoch - best_epoch >= a.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
            break
    head_ready = not head_output or (best_head_error is not None and best_head_error <= 18.0)
    reflection_ready = (not reflection_output or
                        (best_reflection_error is not None and best_reflection_error <= 18.0))
    promoted = head_ready and reflection_ready and (not resumed or
                (best_score >= baseline_score if base_had_reflection
                 else best >= baseline_dice - 0.01))
    if promoted:
        if promoted_path.exists():
            raise FileExistsError(f"Refusing to overwrite {promoted_path}")
        candidate_path.replace(promoted_path)
        print(f"Promoted reflection candidate: Dice {best:.5f}; score {best_score:.5f}")
        print(f"MODEL_OUTPUT={promoted_path.resolve()}")
    else:
        reasons = []
        if not head_ready:
            reasons.append(f"Head error {best_head_error} > 18 px")
        if not reflection_ready:
            reasons.append(f"Reflection error {best_reflection_error} > 18 px")
        if not reasons:
            reasons.append("combined validation score did not improve")
        print(f"Kept selected source model unchanged ({'; '.join(reasons)}); "
              f"candidate retained at {candidate_path}")
    (out / "training_history.json").write_text(json.dumps(
        {"device": device, "train_count": len(train_files), "val_count": len(val_files),
         "best_val_dice": best, "best_epoch": best_epoch, "warm_started": resumed,
         "best_head_error_px": best_head_error,
         "best_reflection_error_px": best_reflection_error,
         "best_selection_score": best_score,
         "baseline_val_dice": baseline_dice, "baseline_head_error_px": baseline_head_error,
         "baseline_reflection_error_px": baseline_reflection_error,
         "baseline_selection_score": baseline_score,
         "candidate_promoted": promoted,
         "model_output": str(promoted_path.resolve()) if promoted else "",
         "candidate_model": str(candidate_path.resolve()) if not promoted else "",
         "base_model": str(base_path.resolve()) if base_path.is_file() else "",
         "requested_lr": requested_lr, "effective_lr": a.lr,
         "head_ready": head_ready, "reflection_ready": reflection_ready,
         "in_channels": in_channels, "head_output": head_output,
         "reflection_output": reflection_output,
         "head_train_labels": sum(has_head_heatmap(root / "heads" / f) for f in train_files),
         "head_val_labels": sum(has_head_heatmap(root / "heads" / f) for f in val_files),
         "reflection_train_labels": sum(has_head_heatmap(root / "reflections" / f) for f in train_files),
         "reflection_val_labels": sum(has_head_heatmap(root / "reflections" / f) for f in val_files),
         "split": "contiguous_temporal_block", "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
