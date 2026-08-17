#!/usr/bin/env python3
"""Train small U-Net on polygon masks; selects best validation Dice checkpoint.

Background-invariant input: when every video in the dataset has a cached
background (unet/preprocess.py, written by prepare_dataset.py), each image
is transformed to x' = 128 + 2*(gray - bg) BEFORE the usual augmentations,
so the static background (walls, floor, lamp) becomes mid-gray and the
network learns "deviations from the background" instead of memorizing one
arena's appearance. The checkpoint records "bg_subtract": true, and
infer.py / head_track.py then apply the identical transform at inference.

Videos without a cached background fall back to raw frames; the checkpoint
flag is only set when ALL videos of the dataset have backgrounds (a mixed
input distribution would silently confuse the model).
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from model import UNet
from preprocess import bg_centered

BG_GAIN = 2.0


def _stem_of(name: str) -> str:
    """image name is {video_stem}_{frame:07d}.png -> video_stem."""
    return name[:-12]  # strip '_' + 7 digits + '.png'


class Pairs(Dataset):
    def __init__(self, root, files, augment=False, backgrounds=None):
        self.root = Path(root); self.files = files; self.augment = augment
        self.backgrounds = backgrounds or {}
        if self.backgrounds:
            probe = cv2.imread(str(self.root / 'images' / files[0]), 0)
            h, w = probe.shape
            self.backgrounds = {s: (cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
                                   if b.shape[:2] != (h, w) else b)
                                for s, b in self.backgrounds.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        x = cv2.imread(str(self.root / 'images' / f), 0)
        y = cv2.imread(str(self.root / 'masks' / f), 0)
        bg = self.backgrounds.get(_stem_of(f))
        if bg is not None:
            x = bg_centered(x, bg, BG_GAIN)
        if self.augment:
            # 90-degree family: covers arena rotations (camera/box turned a
            # quarter-turn) with zero border artifacts - np.rot90 has none.
            k = random.randint(0, 3)
            if k:
                x, y = np.rot90(x, k).copy(), np.rot90(y, k).copy()
            if random.random() < .5:
                x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
            if random.random() < .5:
                x, y = np.flipud(x).copy(), np.flipud(y).copy()
            angle = random.uniform(-18, 18)
            m = cv2.getRotationMatrix2D((x.shape[1] / 2, x.shape[0] / 2), angle, 1)
            x = cv2.warpAffine(x, m, x.shape[::-1], borderMode=cv2.BORDER_REFLECT)
            y = cv2.warpAffine(y, m, y.shape[::-1], flags=cv2.INTER_NEAREST)
            # lighting: affine brightness + non-linear gamma (different lamps,
            # white balance) - gamma models what linear scale+offset cannot.
            # On centered images the mid-gray background level moves with the
            # gamma curve, which is exactly a residual lighting difference.
            x = np.clip(x.astype(np.float32) * random.uniform(.75, 1.25) + random.uniform(-12, 12), 0, 255)
            if random.random() < .6:
                x = 255.0 * (x / 255.0) ** random.uniform(.7, 1.4)
            # sensor noise + mild blur (different camera / focus)
            if random.random() < .5:
                x += np.random.normal(0, random.uniform(1, 6), x.shape)
            if random.random() < .3:
                x = cv2.GaussianBlur(np.clip(x, 0, 255).astype(np.uint8), (3, 3), 0)
            x = np.clip(x, 0, 255).astype(np.uint8)
        return torch.from_numpy(x[None].copy()).float() / 255, torch.from_numpy((y[None] > 127).copy()).float()


def dice(logits, target):
    p = torch.sigmoid(logits)
    return (2 * (p * target).sum() + 1) / ((p + target).sum() + 1)


def load_backgrounds(root: Path, files: list[str]) -> dict[str, np.ndarray]:
    """stem -> bg for every video that has a cached background file."""
    bgs: dict[str, np.ndarray] = {}
    bgdir = root / 'backgrounds'
    for name in files:
        stem = _stem_of(name)
        if stem in bgs:
            continue
        bg = cv2.imread(str(bgdir / f'{stem}.png'), 0)
        if bg is not None:
            bgs[stem] = bg
    return bgs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--epochs', type=int, default=80); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-3); p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    root = Path(a.dataset)
    files = sorted(x.name for x in (root / 'images').glob('*.png'))
    if len(files) < 20:
        raise ValueError(f'Need >=20 masks, found {len(files)}')
    bgs = load_backgrounds(root, files)
    n_videos = len({_stem_of(f) for f in files})
    bg_subtract = len(bgs) == n_videos and n_videos > 0
    if bgs and not bg_subtract:
        print(f'WARNING: {len(bgs)}/{n_videos} videos have a cached background - '
              f'background-invariant input needs ALL of them. Re-run prepare '
              f'(python unet/run_unet.py prepare) once, then retrain. '
              f'Falling back to raw frames for this run.')
    elif bg_subtract:
        print(f'Background-invariant input ON: {n_videos} videos, gain {BG_GAIN:.1f} '
              f'(static background -> mid-gray)')
    else:
        print('No cached backgrounds - training on raw frames (legacy mode).')
    random.shuffle(files)
    n = max(1, round(len(files) * .2))
    train, val = files[n:], files[:n]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lossfn = torch.nn.BCEWithLogitsLoss()
    kw = dict(num_workers=2, pin_memory=dev == 'cuda')
    tl = DataLoader(Pairs(root, train, True, bgs), batch_size=a.batch_size, shuffle=True, **kw)
    vl = DataLoader(Pairs(root, val, False, bgs), batch_size=a.batch_size, **kw)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    best = -1; history = []
    for epoch in range(1, a.epochs + 1):
        model.train(); losses = []
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            z = model(x)
            loss = lossfn(z, y) + (1 - dice(z, y))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval(); scores = []
        with torch.no_grad():
            for x, y in vl:
                scores += [dice(model(x.to(dev)), y.to(dev)).item()]
        score = float(np.mean(scores))
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_dice': score})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            torch.save({'state_dict': model.state_dict(),
                        'size': cv2.imread(str(root / 'images' / files[0]), 0).shape[0],
                        'val_dice': best, 'bg_subtract': bg_subtract, 'bg_gain': BG_GAIN},
                       out / 'best_unet.pt')
    (out / 'training_history.json').write_text(json.dumps(
        {'device': dev, 'train_count': len(train), 'val_count': len(val),
         'best_val_dice': best, 'bg_subtract': bg_subtract, 'bg_gain': BG_GAIN,
         'history': history}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
