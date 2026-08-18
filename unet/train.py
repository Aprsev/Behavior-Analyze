#!/usr/bin/env python3
"""Train the fibre-aware multi-task U-Net from an authoritative manifest.

New models see two channels: raw grayscale keeps a stationary mouse visible;
background residual supplies motion contrast. Existing head labels supervise
an optional heatmap decoder. Training warm-starts from the current best
checkpoint, archives it, and uses temporally grouped validation frames.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import UNet, load_compatible_weights
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
        head = cv2.imread(str(head_path), 0) if head_path.is_file() else np.zeros_like(mask)
        bg = self.backgrounds.get(_stem_of(name))
        if bg is not None and bg.shape != raw.shape:
            bg = cv2.resize(bg, raw.shape[::-1], interpolation=cv2.INTER_AREA)
        x = model_input(raw, bg, self.in_channels, BG_GAIN).transpose(1, 2, 0)
        if x.ndim == 2:
            x = x[..., None]

        if self.augment:
            k = random.randint(0, 3)
            if k:
                x, mask, head = np.rot90(x, k).copy(), np.rot90(mask, k).copy(), np.rot90(head, k).copy()
            if random.random() < .5:
                x, mask, head = np.fliplr(x).copy(), np.fliplr(mask).copy(), np.fliplr(head).copy()
            if random.random() < .5:
                x, mask, head = np.flipud(x).copy(), np.flipud(mask).copy(), np.flipud(head).copy()
            angle = random.uniform(-12, 12)
            matrix = cv2.getRotationMatrix2D((x.shape[1] / 2, x.shape[0] / 2), angle, 1)
            channels = [cv2.warpAffine(x[..., c], matrix, x.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
                        for c in range(x.shape[2])]
            x = np.stack(channels, axis=2)
            mask = cv2.warpAffine(mask, matrix, mask.shape[::-1], flags=cv2.INTER_NEAREST)
            head = cv2.warpAffine(head, matrix, head.shape[::-1], flags=cv2.INTER_LINEAR)
            gain = random.uniform(.88, 1.12); offset = random.uniform(-6, 6)
            x[..., 0] = np.clip(x[..., 0].astype(np.float32) * gain + offset, 0, 255)

        x_t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0
        mask_t = torch.from_numpy((mask[None] > 127).copy()).float()
        head_t = torch.from_numpy((head[None].astype(np.float32) / 255.0).copy())
        head_valid = torch.tensor(float(head.max() > 0), dtype=torch.float32)
        return x_t, mask_t, head_t, head_valid


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


def evaluate(model, loader, device):
    model.eval(); dice_scores = []; head_errors = []
    with torch.no_grad():
        for x, mask, head, valid in loader:
            output = model(x.to(device))
            mask_logits, head_logits = output if isinstance(output, tuple) else (output, None)
            dice_scores.append(float(soft_dice(mask_logits, mask.to(device)).item()))
            if head_logits is not None and valid.any():
                pred = torch.sigmoid(head_logits).cpu().flatten(1).argmax(1)
                true = head.flatten(1).argmax(1)
                width = head.shape[-1]
                distance = torch.sqrt(((pred % width) - (true % width)).float() ** 2 +
                                      ((pred // width) - (true // width)).float() ** 2)
                head_errors.extend(distance[valid > 0].tolist())
    return float(np.mean(dice_scores)), (float(np.mean(head_errors)) if head_errors else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true", help="do not warm-start from existing best_unet.pt")
    p.add_argument("--patience", type=int, default=18)
    a = p.parse_args()
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
    train_files, val_files = temporal_split(files)
    if not train_files or not val_files:
        raise ValueError("Training/validation split is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=in_channels, head_output=head_output).to(device)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best_unet.pt"
    candidate_path = out / "candidate_unet.pt"
    resumed = False
    if best_path.is_file() and not a.fresh:
        old = torch.load(best_path, map_location="cpu")
        source_channel = 1 if bool(old.get("bg_subtract")) else 0
        loaded = load_compatible_weights(model, old["state_dict"], source_channel=source_channel)
        resumed = bool(loaded)
        archive = out / "archive"; archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(best_path, archive / f"best_unet_{stamp}.pt")
        print(f"Warm-started {len(loaded)} tensors; archived previous checkpoint")

    kw = dict(num_workers=2, pin_memory=device == "cuda")
    train_loader = DataLoader(Pairs(root, train_files, True, backgrounds, in_channels),
                              batch_size=a.batch_size, shuffle=True, **kw)
    val_loader = DataLoader(Pairs(root, val_files, False, backgrounds, in_channels),
                            batch_size=a.batch_size, **kw)
    baseline_dice = evaluate(model, val_loader, device)[0] if resumed else -1.0
    if resumed:
        print(f"Current checkpoint Dice on this exact temporal validation split: {baseline_dice:.5f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=(min(a.lr, 2e-4) if resumed else a.lr),
                                  weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    best = -1.0; best_epoch = 0; history = []
    for epoch in range(1, a.epochs + 1):
        model.train(); losses = []
        for x, mask, head, valid in train_loader:
            x, mask, head, valid = x.to(device), mask.to(device), head.to(device), valid.to(device)
            output = model(x)
            mask_logits, head_logits = output if isinstance(output, tuple) else (output, None)
            loss = (bce(mask_logits, mask) + (1.0 - soft_dice(mask_logits, mask)) +
                    0.25 * hard_negative_loss(mask_logits, mask))
            if head_logits is not None and valid.any():
                # A head Gaussian occupies very few pixels; unweighted BCE
                # would minimize loss by predicting zero everywhere.
                head_prob = torch.sigmoid(head_logits)
                per_sample = (((head_prob - head) ** 2) * (1.0 + 24.0 * head)).mean((1, 2, 3))
                loss = loss + 0.30 * (per_sample * valid).sum() / valid.sum().clamp_min(1)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        val_dice, head_error = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": val_dice,
               "head_error_px": head_error}
        history.append(row); print(json.dumps(row), flush=True)
        if val_dice > best:
            best, best_epoch = val_dice, epoch
            torch.save({"state_dict": model.state_dict(), "size": sample_size,
                        "val_dice": best, "in_channels": in_channels,
                        "dual_channel": True, "bg_subtract": True, "bg_gain": BG_GAIN,
                        "head_output": head_output, "checkpoint_version": 2,
                        "validation": "contiguous_temporal_block"}, candidate_path)
        if epoch - best_epoch >= a.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
            break
    promoted = best >= baseline_dice
    if promoted:
        shutil.copy2(candidate_path, best_path)
        print(f"Promoted candidate: Dice {best:.5f} >= baseline {baseline_dice:.5f}")
    else:
        print(f"Kept previous best model: candidate Dice {best:.5f} < baseline {baseline_dice:.5f}")
    (out / "training_history.json").write_text(json.dumps(
        {"device": device, "train_count": len(train_files), "val_count": len(val_files),
         "best_val_dice": best, "best_epoch": best_epoch, "warm_started": resumed,
         "baseline_val_dice": baseline_dice, "candidate_promoted": promoted,
         "in_channels": in_channels, "head_output": head_output,
         "split": "contiguous_temporal_block", "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
