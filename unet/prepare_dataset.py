#!/usr/bin/env python3
"""Convert saved irregular torso polygons into images/masks for U-Net.

Supports multiple videos: run once per video with the same --output-dir and
--labels; image names are prefixed with the video stem so frames never
collide, and dataset.json accumulates one entry per video. Pass
--exclude-csv (the screening CSV from unet/screen_frames.py) to drop frames
manually flagged during screening (mouse absent, human intervention, ...).

Every video also gets a cached background (unet/preprocess.py) saved as
<output-dir>/backgrounds/<stem>.png; train.py uses it to feed the network
background-centered images, which removes inter-video background differences.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from preprocess import estimate_background, save_background

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "traditional" / "code"
sys.path.insert(0, str(CODE))
from mouse_behavior_pipeline import perspective_geometry  # noqa: E402


def gaussian_heatmap(size: int, x: float, y: float, sigma: float = 4.0) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
    return np.clip(heat * 255.0, 0, 255).astype(np.uint8)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True); p.add_argument('--labels', required=True)
    p.add_argument('--heads', default='', help='optional head_anchor_calibration.csv')
    p.add_argument('--roi-json', default='', help='ROI needed to convert head cm coordinates to source pixels')
    p.add_argument('--arena-width-cm', type=float, default=25.0)
    p.add_argument('--arena-height-cm', type=float, default=30.0)
    p.add_argument('--output-dir', required=True); p.add_argument('--size', type=int, default=256)
    p.add_argument('--exclude-csv', default=''); a = p.parse_args()
    labels = pd.read_csv(a.labels); labels = labels.loc[~labels.exclude.fillna(False).astype(bool)]
    if 'polygon_px' not in labels:
        raise ValueError('labels must be polygon-based manual_torso_constraints.csv')
    if 'video' in labels.columns:
        # Multi-video labels: each row belongs to one recording; the same frame
        # number in another video is a different image, so filter strictly.
        labels = labels.loc[labels.video == Path(a.video).name]
        print(f'Using {len(labels)} labels for {Path(a.video).name}')
    excluded = set()
    if a.exclude_csv and Path(a.exclude_csv).is_file():
        ex = pd.read_csv(a.exclude_csv)
        ex = ex.loc[ex.exclude.fillna(False).astype(bool) & (ex.video == Path(a.video).name)]
        excluded = set(int(f) for f in ex.frame)
        print(f'Excluding {len(excluded)} screened frames of {Path(a.video).name}')
    out = Path(a.output_dir); images = out / 'images'; masks = out / 'masks'; head_dir = out / 'heads'
    images.mkdir(parents=True, exist_ok=True); masks.mkdir(parents=True, exist_ok=True)
    head_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.video).stem.replace(' ', '_')
    heads = pd.DataFrame()
    if a.heads and Path(a.heads).is_file():
        heads = pd.read_csv(a.heads)
        if 'video' in heads.columns:
            heads = heads.loc[heads.video == Path(a.video).name]
        if 'exclude' in heads.columns:
            heads = heads.loc[~heads.exclude.fillna(False).astype(bool)]
        if len(heads):
            heads = heads.drop_duplicates('frame', keep='last').set_index('frame')
    inverse = None; rect_w = rect_h = None
    if a.roi_json and Path(a.roi_json).is_file():
        roi = json.loads(Path(a.roi_json).read_text(encoding='utf-8'))
        corners = np.asarray(roi['arena_corners_px'], np.float32)
        rect_w, rect_h, _, inverse, _, _ = perspective_geometry(
            corners, a.arena_width_cm, a.arena_height_cm)
    cap = cv2.VideoCapture(a.video); written = []
    for _, row in labels.iterrows():
        frame_index = int(row.frame)
        if frame_index in excluded or not isinstance(row.polygon_px, str) or not row.polygon_px:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index); ok, frame = cap.read()
        if not ok: continue
        polygon = np.asarray(json.loads(row.polygon_px), np.float32)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (a.size, a.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (a.size, a.size), interpolation=cv2.INTER_NEAREST)
        name = f'{stem}_{frame_index:07d}.png'
        cv2.imwrite(str(images / name), gray)
        cv2.imwrite(str(masks / name), mask)
        # Existing manual head points now contribute to training.  They are
        # stored in arena cm, converted through the same ROI calibration back
        # to source pixels, then represented as a smooth heatmap.
        head_map = np.zeros((a.size, a.size), np.uint8)
        if inverse is not None and frame_index in heads.index:
            hr = heads.loc[frame_index]
            hx = pd.to_numeric(hr.get('head_x_cm'), errors='coerce')
            hy = pd.to_numeric(hr.get('head_y_cm'), errors='coerce')
            present = bool(hr.get('head_present', True))
            if present and np.isfinite(hx) and np.isfinite(hy):
                rect_pt = np.asarray([[[float(hx) / a.arena_width_cm * (rect_w - 1),
                                        float(hy) / a.arena_height_cm * (rect_h - 1)]]], np.float32)
                sx, sy = cv2.perspectiveTransform(rect_pt, inverse)[0, 0]
                px = float(sx) * a.size / frame.shape[1]
                py = float(sy) * a.size / frame.shape[0]
                if 0 <= px < a.size and 0 <= py < a.size:
                    head_map = gaussian_heatmap(a.size, px, py)
        cv2.imwrite(str(head_dir / name), head_map)
        written.append(frame_index)
    cap.release()

    # Synchronize this video's files with the authoritative CSV.  Previously
    # excluded/deleted annotations used to remain as orphan PNGs and train.py
    # silently kept training on them.
    wanted = {f'{stem}_{frame_index:07d}.png' for frame_index in written}
    removed = 0
    for folder in (images, masks, head_dir):
        for path in folder.glob(f'{stem}_???????.png'):
            if path.name not in wanted and path.is_file():
                path.unlink()
                removed += 1
    if removed:
        print(f'Removed {removed} stale image/mask/head files for {Path(a.video).name}')
    # Background cache: removes inter-video background differences at train
    # time (train.py only uses it when every video in the dataset has one).
    bg_path = out / 'backgrounds' / f'{stem}.png'
    bg = estimate_background(a.video)
    if bg is not None:
        bg_path.parent.mkdir(parents=True, exist_ok=True)
        save_background(bg_path, bg)
        print(f'Cached background {bg_path.name} for background-invariant training')
    else:
        print(f'WARNING: background estimation failed for {Path(a.video).name}; '
              f'train.py will use raw frames for this video')
    manifest = {'video': str(Path(a.video).resolve()), 'count': len(written), 'size': a.size,
                'frames': written, 'samples': sorted(wanted), 'stem': stem,
                'head_count': int(sum(cv2.imread(str(head_dir / name), 0).max() > 0 for name in wanted)),
                'background': str(bg_path) if bg is not None else ''}
    if (out / 'dataset.json').exists():
        old = json.loads((out / 'dataset.json').read_text(encoding='utf-8'))
        videos = old.get('videos') or []
        videos = [v for v in videos if v.get('video') != manifest['video']]
        videos.append(manifest)
    else:
        videos = [manifest]
    (out / 'dataset.json').write_text(json.dumps({'size': a.size, 'videos': videos}, indent=2), encoding='utf-8')
    print(f'Wrote {len(written)} image/mask pairs from {Path(a.video).name} to {out} (total videos: {len(videos)})')

if __name__ == '__main__':
    main()
