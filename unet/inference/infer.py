#!/usr/bin/env python3
"""Run U-Net segmentation; export mask/overlay videos and CNN-centroid CSV.

Pass --exclude-csv (the screening CSV from unet/screen_frames.py) to make
manually excluded frames visible in the outputs: their trajectory rows become
NaN, the mask video shows a black frame, and the overlay video marks them with
a red EXCLUDED label. The excluded frames are also reported in inference.json.

Pass --rotate 90/180/270 when the arena was turned by a quarter-turn compared
to the training videos: frames are rotated before the CNN and the mask /
centroid are rotated back, so the model always sees the training orientation.

Background-invariant inference: when the checkpoint was trained with
background subtraction ("bg_subtract" in the .pt file, see train.py), the
per-video background is estimated here with the very same function used at
training time and the frame is centered on it before the CNN (background ->
mid-gray). Old checkpoints without the flag keep the raw-input behaviour.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from core.model import checkpoint_model
from core.preprocess import estimate_background
from core.postprocess import TemporalMaskFilter, model_input

def rot_pt(p, k, h, w):
    """(col,row) in an (h,w) frame -> (col,row) in np.rot90(frame,k)."""
    x, y = p
    for _ in range(k % 4):
        x, y = y, w - 1 - x; h, w = w, h
    return x, y

def inv_rot_pt(p, k, h, w):
    """inverse of rot_pt (p is in the rotated frame)."""
    x, y = p
    for _ in range(k % 4): h, w = w, h
    for _ in range(k % 4):
        x, y = h - 1 - y, x; h, w = w, h
    return x, y

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True); p.add_argument('--model', required=True)
    p.add_argument('--output-dir', required=True); p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    p.add_argument('--fibre-opening', type=int, default=5)
    p.add_argument('--reacquire-sec', type=float, default=.35)
    p.add_argument('--exclude-csv', default=''); a = p.parse_args()
    excluded = set()
    if a.exclude_csv and Path(a.exclude_csv).is_file():
        ex = pd.read_csv(a.exclude_csv)
        ex = ex.loc[ex.exclude.fillna(False).astype(bool) & (ex.video == Path(a.video).name)]
        excluded = set(int(f) for f in ex.frame)
        print(f'Marking {len(excluded)} screened frames as excluded')
    pack = torch.load(a.model, map_location='cpu')
    size = int(pack['size']); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_channels = int(pack.get('in_channels', 2 if pack.get('dual_channel') else 1))
    net = checkpoint_model(pack, dev); net.eval()
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS); w, h = int(cap.get(3)), int(cap.get(4))
    k = int(a.rotate) // 90 % 4
    # Background-invariant preprocessing, identical to train.py: only active
    # when the model was trained with it, so old checkpoints are unaffected.
    bg_small = None
    if bool(pack.get('bg_subtract')) or in_channels > 1:
        bg = estimate_background(a.video)
        if bg is not None:
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            if k:
                bg_gray = np.rot90(bg_gray, k)
            bg_small = cv2.resize(bg_gray, (size, size), interpolation=cv2.INTER_AREA)
            print(f'Model trained with background subtraction - applying '
                  f'background-centered input (gain {pack.get("bg_gain", 2.0):.1f})')
        else:
            print('WARNING: background estimation failed; using raw frames')
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mw = cv2.VideoWriter(str(out / 'mouse_miniscope_mask.mp4'), fourcc, fps, (w, h))
    ow = cv2.VideoWriter(str(out / 'mouse_miniscope_overlay.mp4'), fourcc, fps, (w, h))
    temporal = TemporalMaskFilter(fps=fps, opening_px=a.fibre_opening, hold_frames=3,
                                  reacquire_frames=max(8, round(fps * a.reacquire_sec)))
    rows = []; i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i in excluded:
            clean = np.zeros((h, w), np.uint8); cx = cy = float('nan')
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)
            cv2.putText(overlay, f'EXCLUDED frame {i}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)
        else:
            frame_in = np.rot90(frame, k) if k else frame
            hi, wi = frame_in.shape[:2]
            gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (size, size))
            channels = model_input(small, bg_small, in_channels, float(pack.get('bg_gain', 2.0)))
            x = torch.from_numpy(channels[None].copy()).float().to(dev) / 255
            with torch.no_grad():
                output = net(x)
                mask_logits = output[0] if isinstance(output, tuple) else output
                prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
            result = temporal.update(prob, a.threshold)
            if result.centroid is not None:
                clean = cv2.resize(result.mask, (wi, hi), interpolation=cv2.INTER_NEAREST)
                cx = result.centroid[0] * wi / size
                cy = result.centroid[1] * hi / size
                if k:
                    cx, cy = inv_rot_pt((float(cx), float(cy)), k, h, w)
            else:
                clean = np.zeros((hi, wi), np.uint8); cx = cy = np.nan
            if k:
                clean = np.rot90(clean, -k)
            overlay = frame.copy(); overlay[clean > 0] = (0, 220, 0)
            overlay = cv2.addWeighted(frame, .65, overlay, .35, 0)
        mw.write(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)); ow.write(overlay)
        rows.append((i, i / fps, cx, cy)); i += 1
    cap.release(); mw.release(); ow.release()
    pd.DataFrame(rows, columns=['frame', 'timestamp_sec', 'body_x_px', 'body_y_px']).to_csv(
        out / 'unet_trajectory.csv', index=False)
    manifest = {'input': str(Path(a.video).resolve()), 'model': str(Path(a.model).resolve()),
                'device': dev, 'frames': i, 'threshold': a.threshold, 'rotate': a.rotate,
                'in_channels': in_channels, 'fibre_aware_temporal_filter': True,
                'fibre_opening': a.fibre_opening, 'reacquire_sec': a.reacquire_sec,
                'bg_subtract': bg_small is not None,
                'excluded_frames': sorted(excluded & set(range(i))),
                'excluded_count': len(excluded & set(range(i)))}
    (out / 'inference.json').write_text(json.dumps(manifest, indent=2))
    print(f'Wrote {i} frames to {out} (excluded {manifest["excluded_count"]})')

if __name__ == '__main__':
    main()
