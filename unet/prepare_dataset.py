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
from label_compat import as_bool, normalize_polygon, video_mask

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "traditional" / "code"
sys.path.insert(0, str(CODE))
from mouse_behavior_pipeline import perspective_geometry  # noqa: E402


def gaussian_heatmap(size: int, x: float, y: float, sigma: float = 4.0) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
    return np.clip(heat * 255.0, 0, 255).astype(np.uint8)


def has_head_heatmap(path: Path) -> bool:
    """Return False for an absent/corrupt/empty optional head target."""
    if not path.is_file():
        return False
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image is not None and bool(image.max() > 0)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True); p.add_argument('--labels', required=True)
    p.add_argument('--heads', default='', help='optional head_anchor_calibration.csv')
    p.add_argument('--roi-json', default='', help='ROI needed to convert head cm coordinates to source pixels')
    p.add_argument('--arena-width-cm', type=float, default=25.0)
    p.add_argument('--arena-height-cm', type=float, default=30.0)
    p.add_argument('--max-head-reflection-cm', type=float, default=3.0,
                   help='legacy unverified reflections farther from Head are rejected')
    p.add_argument('--output-dir', required=True); p.add_argument('--size', type=int, default=256)
    p.add_argument('--exclude-csv', default=''); a = p.parse_args()
    labels = pd.read_csv(a.labels)
    if 'polygon_px' not in labels:
        raise ValueError('labels must be polygon-based manual_torso_constraints.csv')
    if 'video' in labels.columns:
        # Multi-video labels: each row belongs to one recording; the same frame
        # number in another video is a different image, so filter strictly.
        labels = labels.loc[video_mask(labels.video, Path(a.video).name)]
    labels = labels.drop_duplicates('frame', keep='last')
    matched_label_count = len(labels)
    label_excluded = set(labels.loc[labels.exclude.map(as_bool), 'frame'].astype(int)) if 'exclude' in labels else set()
    excluded = set(label_excluded)
    screening_excluded = set()
    if a.exclude_csv and Path(a.exclude_csv).is_file():
        ex = pd.read_csv(a.exclude_csv)
        matches = video_mask(ex.video, Path(a.video).name) if 'video' in ex else pd.Series(True, index=ex.index)
        ex = ex.loc[ex.exclude.map(as_bool) & matches]
        screening_excluded = set(int(f) for f in ex.frame)
        excluded.update(screening_excluded)
    labels = labels.loc[~labels.frame.astype(int).isin(excluded)]
    print(f'Label audit for {Path(a.video).name}: matched={matched_label_count}, '
          f'usable={len(labels)}, excluded_union={len(excluded)}, '
          f'label_excluded={len(label_excluded)}, screening_excluded={len(screening_excluded)}')
    out = Path(a.output_dir); images = out / 'images'; masks = out / 'masks'
    head_dir = out / 'heads'; reflection_dir = out / 'reflections'
    images.mkdir(parents=True, exist_ok=True); masks.mkdir(parents=True, exist_ok=True)
    head_dir.mkdir(parents=True, exist_ok=True); reflection_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.video).stem.replace(' ', '_')
    heads = pd.DataFrame()
    if a.heads and Path(a.heads).is_file():
        heads = pd.read_csv(a.heads)
        if 'video' in heads.columns:
            heads = heads.loc[video_mask(heads.video, Path(a.video).name)]
        if 'exclude' in heads.columns:
            heads = heads.loc[~heads.exclude.map(as_bool)]
        if len(heads):
            heads = heads.drop_duplicates('frame', keep='last').set_index('frame')
            head_xy = heads.reindex(columns=['head_x_cm', 'head_y_cm']).apply(
                pd.to_numeric, errors='coerce').to_numpy(float)
            reflection_xy = heads.reindex(columns=['reflection_x_cm', 'reflection_y_cm']).apply(
                pd.to_numeric, errors='coerce').to_numpy(float)
            head_present = (heads['head_present'].map(as_bool).to_numpy()
                            if 'head_present' in heads else np.isfinite(head_xy).all(axis=1))
            if 'head_verified' in heads:
                head_verified_raw = heads['head_verified']
                head_trusted = head_verified_raw.isna().to_numpy() | head_verified_raw.map(as_bool).to_numpy()
            else:
                head_trusted = np.ones(len(heads), dtype=bool)  # legacy manual rows
            heads['_head_usable'] = head_present & head_trusted
            both = (heads['_head_usable'].to_numpy(bool) &
                    np.isfinite(head_xy).all(axis=1) & np.isfinite(reflection_xy).all(axis=1))
            distance = np.full(len(heads), np.inf, dtype=float)
            distance[both] = np.linalg.norm(head_xy[both] - reflection_xy[both], axis=1)
            verified = (heads['reflection_verified'].map(as_bool).to_numpy()
                        if 'reflection_verified' in heads else np.zeros(len(heads), dtype=bool))
            reflection_present = (heads['reflection_present'].map(as_bool).to_numpy()
                                  if 'reflection_present' in heads else
                                  np.isfinite(reflection_xy).all(axis=1))
            heads['_reflection_usable'] = reflection_present & (
                verified | (both & (distance <= a.max_head_reflection_cm)))
            reflection_rejected = int((reflection_present & ~heads['_reflection_usable'].to_numpy(bool)).sum())
            print(f'Reflection label audit: accepted={int(heads["_reflection_usable"].sum())}, '
                  f'rejected_tail_or_unverified={reflection_rejected}, '
                  f'max_head_distance_cm={a.max_head_reflection_cm:.1f}')
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
        try:
            polygon = normalize_polygon(row.polygon_px)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f'WARNING: skipped invalid polygon frame {frame_index}: {exc}')
            continue
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
        reflection_map = np.zeros((a.size, a.size), np.uint8)
        if inverse is not None and frame_index in heads.index:
            hr = heads.loc[frame_index]
            def labelled_map(prefix: str, present_column: str) -> np.ndarray:
                x_cm = pd.to_numeric(hr.get(f'{prefix}_x_cm'), errors='coerce')
                y_cm = pd.to_numeric(hr.get(f'{prefix}_y_cm'), errors='coerce')
                present = as_bool(hr.get(present_column, True))
                if prefix == 'head' and not as_bool(hr.get('_head_usable', False)):
                    present = False
                if prefix == 'reflection' and not as_bool(hr.get('_reflection_usable', False)):
                    present = False
                if not present or not np.isfinite([x_cm, y_cm]).all():
                    return np.zeros((a.size, a.size), np.uint8)
                rect_pt = np.asarray([[[float(x_cm) / a.arena_width_cm * (rect_w - 1),
                                        float(y_cm) / a.arena_height_cm * (rect_h - 1)]]], np.float32)
                sx, sy = cv2.perspectiveTransform(rect_pt, inverse)[0, 0]
                px = float(sx) * a.size / frame.shape[1]
                py = float(sy) * a.size / frame.shape[0]
                if 0 <= px < a.size and 0 <= py < a.size:
                    return gaussian_heatmap(a.size, px, py)
                return np.zeros((a.size, a.size), np.uint8)
            head_map = labelled_map('head', 'head_present')
            reflection_map = labelled_map('reflection', 'reflection_present')
        cv2.imwrite(str(head_dir / name), head_map)
        cv2.imwrite(str(reflection_dir / name), reflection_map)
        written.append(frame_index)
    cap.release()

    # Synchronize this video's files with the authoritative CSV.  Previously
    # excluded/deleted annotations used to remain as orphan PNGs and train.py
    # silently kept training on them.
    wanted = {f'{stem}_{frame_index:07d}.png' for frame_index in written}
    removed = 0
    for folder in (images, masks, head_dir, reflection_dir):
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
    head_count = int(sum(has_head_heatmap(head_dir / name) for name in wanted))
    reflection_count = int(sum(has_head_heatmap(reflection_dir / name) for name in wanted))
    manifest = {'video': str(Path(a.video).resolve()), 'count': len(written), 'size': a.size,
                'frames': written, 'samples': sorted(wanted), 'stem': stem,
                'source_label_count': matched_label_count,
                'label_excluded_frames': sorted(label_excluded),
                'screening_excluded_frames': sorted(screening_excluded),
                'excluded_frames': sorted(excluded),
                'head_count': head_count,
                'reflection_count': reflection_count,
                'background': str(bg_path) if bg is not None else ''}
    if (out / 'dataset.json').exists():
        old = json.loads((out / 'dataset.json').read_text(encoding='utf-8'))
        videos = old.get('videos') or []
        # Replace by stable stem as well as absolute path. This prevents an old
        # host path from retaining excluded samples after project migration.
        videos = [v for v in videos if v.get('video') != manifest['video'] and v.get('stem') != stem]
        videos.append(manifest)
    else:
        videos = [manifest]
    (out / 'dataset.json').write_text(json.dumps({'size': a.size, 'videos': videos}, indent=2), encoding='utf-8')
    print(f'Wrote {len(written)} image/mask pairs from {Path(a.video).name} to {out} '
          f'(head labels {head_count}; reflection labels {reflection_count}; total videos: {len(videos)})')

if __name__ == '__main__':
    main()
