#!/usr/bin/env python3
"""Adjust automatically proposed mouse contours as irregular polygons.

Each selected frame starts with an automatic body contour. Drag its cyan
vertices to exclude the fibre/tail while retaining mouse + miniscope. The
saved polygon is an exact frame-level segmentation constraint, not a box.

Frame selection is maximally diverse: pass --candidate-csv (the screening
CSV from unet/screen_frames.py) to label exactly the screened, junk-free
candidates; otherwise farthest-point sampling over arena descriptors picks
the most different frames of the video. Already-labelled frames are skipped.

Controls: drag cyan vertex; click edge to add vertex; Backspace removes nearest
vertex; R restores automatic contour; E toggles exclusion; A/D step; J/L +/-10;
S save; Q/Esc save and quit.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
UNET = Path(__file__).resolve().parents[2] / 'unet'
sys.path.insert(0, str(UNET))
from core.label_compat import as_bool, normalize_polygon, video_mask  # noqa: E402
from annotation.similarity_propagation import (  # noqa: E402
    read_frame, similar_candidate_neighbors, translate_polygon_optical_flow)
from mouse_behavior_pipeline import (
    perspective_geometry, robust_threshold, sample_frames, segment_mouse, video_properties,
)


def simplify(points: np.ndarray, count: int = 8) -> np.ndarray:
    if len(points) <= count: return points.astype(float)
    return points[np.linspace(0, len(points)-1, count, dtype=int)].astype(float)


def scan_candidates(
    cap: cv2.VideoCapture, total: int, forward: np.ndarray, rw: int, rh: int,
    background: np.ndarray, threshold: float, pool_cap: int = 2000,
    excluded: set = frozenset(), suspicious_area: float = 0.0,
) -> tuple[list, list]:
    """Scan a uniform grid of frames; return (good, junk) candidate lists.

    good: (frame, normalized 48x48 arena descriptor, rectified torso contour)
    junk: (frame, None) where the automatic segmentation failed, the mask
    size is implausible (mouse absent, human intervention, severe occlusion,
    motion blur, ...), or - with suspicious_area > 0 - the foreground
    difference covers an unexpectedly large share of the arena (a hand or
    object entering the arena often evades the component filter).
    """
    step = max(1, total // pool_cap)
    good, junk = [], []
    # Sequential reading: cap.set() seeks are unreliable on MJPG (frame offset
    # accumulates over hundreds of seeks), which would silently sample wrong
    # frames. Reading frame by frame costs little and cannot misalign.
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % step == 0 and f not in excluded:
            rect = cv2.warpPerspective(frame, forward, (rw, rh))
            pixels = rect.shape[0] * rect.shape[1]  # single-channel arena size
            if suspicious_area > 0:
                diff = np.max(cv2.subtract(background, rect), axis=2)
                if float(np.count_nonzero(diff > threshold)) > suspicious_area * pixels:
                    junk.append((f, None))
                    f += 1
                    continue
            detection = segment_mouse(rect, background, threshold, None, float('inf'), None)
            if detection.contour is None or not (0.001 * pixels <= detection.area <= 0.2 * pixels):
                junk.append((f, None))
                f += 1
                continue
            gray = cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (48, 48)).astype(np.float32)
            gray = (gray - gray.mean()) / (gray.std() + 1e-6)
            good.append((f, gray.ravel(), detection.contour))
        f += 1
    return good, junk


def farthest_pick(descriptors: np.ndarray, count: int) -> list[int]:
    """Greedy farthest-point sampling: pick frames whose content (position,
    posture, appearance) differs most from everything already picked."""
    if not len(descriptors):
        return []
    rng = np.random.default_rng()
    picked = [int(rng.integers(len(descriptors)))]
    mind = np.linalg.norm(descriptors - descriptors[picked[0]], axis=1)
    while len(picked) < min(count, len(descriptors)):
        j = int(np.argmax(mind)); picked.append(j)
        mind = np.minimum(mind, np.linalg.norm(descriptors - descriptors[j], axis=1))
    return picked


def pick_diverse_frames(
    cap: cv2.VideoCapture, total: int, forward: np.ndarray, rw: int, rh: int,
    background: np.ndarray, threshold: float, count: int,
    excluded: set = frozenset(),
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Quality-filtered farthest-point frame selection across the whole video."""
    good, _ = scan_candidates(cap, total, forward, rw, rh, background, threshold, 2000, excluded)
    if not good:
        return []
    indices = farthest_pick(np.stack([d for _, d, _ in good]), count)
    return [good[i] for i in indices]


def _select_frames(args, labels: dict, cap: cv2.VideoCapture, total: int,
                   forward: np.ndarray, rw: int, rh: int,
                   background: np.ndarray, threshold: float) -> list[int]:
    """Use the manually screened candidate CSV (diverse + junk removed).

    When --candidate-csv is given but unusable (missing file, or no rows for
    this video) the tool refuses instead of silently falling back to automatic
    picking: an automatic fallback would reintroduce exactly the invalid
    frames (mouse absent, human intervention) the screening pass removes.
    Only run farthest-point sampling when no candidate CSV is requested.
    """
    if args.candidate_csv:
        if not Path(args.candidate_csv).is_file():
            raise SystemExit(f'--candidate-csv {args.candidate_csv} not found.\n'
                             f'Run screening first: python unet/run_unet.py screen')
        cand = pd.read_csv(args.candidate_csv)
        cand = cand.loc[video_mask(cand.video, Path(args.input).name)]
        if not len(cand):
            raise SystemExit(f'No screening rows for {Path(args.input).name}.\n'
                             f'Run screening first: python unet/run_unet.py screen')
        keep = cand.loc[~cand.exclude.map(as_bool)].frame.astype(int)
        frames = [f for f in dict.fromkeys(int(f) for f in keep)
                  if args.only_modified or f not in labels]
        print(f'Using {len(frames)} screened candidate frames (junk already removed).')
        return frames[:args.max_labels]
    picks = pick_diverse_frames(cap, total, forward, rw, rh, background, threshold, args.max_labels, set())
    return [f for f, _, _ in picks if f not in labels]


def merge_labels(path: Path, video_name: str, rows: list[dict]) -> None:
    """Upsert this video's rows into the shared labels CSV.

    The CSV holds annotations for EVERY video. A naive write of just this
    video's rows would silently delete all other videos' annotations the
    moment a second video is annotated - the classic "my old annotations
    disappeared" bug. Only the (video, frame) pairs touched in this session
    are replaced; every other row is kept byte-for-byte.
    """
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["frame", "polygon_px", "exclude"])
    if "video" not in old.columns:
        old["video"] = video_name  # legacy single-video CSV: all rows are ours
    else:
        old.loc[video_mask(old.video, video_name), "video"] = video_name
    touched = [r["frame"] for r in rows]
    keep = ~((old.video == video_name) & old.frame.isin(touched))
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat([old.loc[keep], pd.DataFrame(rows)], ignore_index=True) \
        .drop_duplicates(["video", "frame"], keep="last") \
        .sort_values(["video", "frame"])
    # Write-verify-replace prevents a closed/crashed GUI from leaving a
    # partially written CSV. Keep the immediately previous version as .bak.
    temporary = path.with_suffix(path.suffix + ".tmp")
    backup = path.with_suffix(path.suffix + ".bak")
    merged.to_csv(temporary, index=False)
    check = pd.read_csv(temporary)
    expected = {(video_name, int(row["frame"])) for row in rows}
    actual = set(zip(check.video.astype(str), check.frame.astype(int)))
    missing = expected - actual
    if missing:
        temporary.unlink(missing_ok=True)
        raise IOError(f"Save verification failed; {len(missing)} annotations are missing")
    if path.is_file():
        shutil.copy2(path, backup)
    temporary.replace(path)
    print(f"Saved and verified {len(rows)} polygon torso constraints to {path}")
    if backup.is_file():
        print(f"Previous label file backed up as {backup}")


def merge_annotation_exclusions(screening_csv: str, video_name: str, label_rows: list[dict]) -> None:
    """Record frames excluded during annotation (E key) in the screening CSV,
    so prepare/infer also skip them. Silently skips when no screening CSV."""
    if not screening_csv or not Path(screening_csv).is_file():
        return
    path = Path(screening_csv)
    df = pd.read_csv(path)
    values = {int(row['frame']): as_bool(row.get('exclude')) for row in label_rows}
    for frame, excluded in values.items():
        matches = video_mask(df.video, video_name) if 'video' in df else pd.Series(True, index=df.index)
        df.loc[matches & (df.frame == frame), 'exclude'] = int(excluded)
    df.to_csv(path, index=False)
    print(f'Synchronized exclusion state for {len(values)} annotations in {screening_csv}')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--comparison-csv',required=True)
    p.add_argument('--roi-json',required=True); p.add_argument('--output',required=True)
    p.add_argument('--arena-width-cm',type=float,required=True); p.add_argument('--arena-height-cm',type=float,required=True)
    p.add_argument('--max-labels',type=int,default=50); p.add_argument('--candidate-csv',default='')
    p.add_argument('--mask-video', default='',
                   help='optional mask video used as the initial editable U-Net contour')
    p.add_argument('--only-modified', action='store_true',
                   help='write only frames whose polygon was actually edited')
    p.add_argument('--propagate-similarity', type=float, default=0.0,
                   help='auto-propagate a manual edit to adjacent candidates above this similarity')
    p.add_argument('--propagate-window', type=int, default=15)
    p.add_argument('--review-existing', action='store_true',
                   help='review and overwrite existing labels for this video instead of selecting new frames')
    args=p.parse_args()
    data=pd.read_csv(args.comparison_csv)
    old=pd.read_csv(args.output) if Path(args.output).exists() else pd.DataFrame(columns=['frame','polygon_px','exclude'])
    if 'video' in old.columns:
        # Multi-video labels: only rows belonging to THIS video are valid;
        # the same frame number in another video is a different recording.
        old = old.loc[video_mask(old.video, Path(args.input).name)]
    labels={int(row.frame):row for _,row in old.iterrows()}
    corners=np.asarray(json.loads(Path(args.roi_json).read_text(encoding='utf-8'))['arena_corners_px'],np.float32)
    total,fps,_,_=video_properties(Path(args.input))
    rw,rh,forward,inverse,_,_=perspective_geometry(corners,args.arena_width_cm,args.arena_height_cm)
    _,samples=sample_frames(Path(args.input),total,61)
    rect_samples=np.stack([cv2.warpPerspective(frame,forward,(rw,rh)) for frame in samples])
    background=np.percentile(rect_samples,85,axis=0).astype(np.uint8)
    threshold,_=robust_threshold(rect_samples,background,0)
    cap=cv2.VideoCapture(args.input)
    mask_cap=(cv2.VideoCapture(args.mask_video)
              if args.mask_video and Path(args.mask_video).is_file() else None)
    title='Polygon torso correction (cyan); remove fibre/tail'
    if args.review_existing:
        frames = sorted(frame for frame, row in labels.items()
                        if isinstance(row.get('polygon_px'), str) and row.get('polygon_px'))
        if args.max_labels > 0:
            frames = frames[:args.max_labels]
        print(f'Review mode: loaded {len(frames)} existing labels for {Path(args.input).name}.')
    else:
        frames=_select_frames(args,labels,cap,total,forward,rw,rh,background,threshold)
    if not frames:
        print('No existing labels to review.' if args.review_existing else
              'No new frames to label: all screened candidates are already labelled.')
        cap.release(); return
    if not args.review_existing:
        print(f'{Path(args.input).name}: {len(labels)} frames already labelled, '
              f'{len(frames)} NEW unlabelled frames to label (max {args.max_labels} per run).')
    state={'i':0,'points':None,'drag':None,'dirty':False,'touched':set(),
           'propagated':0}
    def frame_image(frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,img=cap.read()
        if not ok: raise RuntimeError(f'Cannot read frame {frame}')
        return img
    def auto_polygon(frame):
        # The same dark-foreground / compact-torso segmentation used in the
        # tracking pipeline supplies the initial editable contour.
        image=frame_image(frame)
        if mask_cap is not None:
            mask_cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,mask_frame=mask_cap.read()
            if ok:
                gray=cv2.cvtColor(mask_frame,cv2.COLOR_BGR2GRAY)
                contours,_=cv2.findContours((gray>127).astype(np.uint8),cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                if contours:
                    contour=max(contours,key=cv2.contourArea).reshape(-1,2).astype(float)
                    contour[:,0]*=image.shape[1]/gray.shape[1]
                    contour[:,1]*=image.shape[0]/gray.shape[0]
                    return simplify(contour,8)
        rectified=cv2.warpPerspective(image,forward,(rw,rh))
        detection=segment_mouse(rectified,background,threshold,None,float('inf'),None)
        if detection.contour is not None:
            source_contour=cv2.perspectiveTransform(detection.contour.astype(np.float32),inverse)[:,0,:]
            return simplify(source_contour,8)
        row=data.loc[data.frame==frame].iloc[0]
        # Fallback only when automatic segmentation failed entirely.
        cx,cy=row.body_x_cm/args.arena_width_cm*(rw-1),row.body_y_cm/args.arena_height_cm*(rh-1)
        pts=[]
        for x,y in ((row.head_silhouette_x_cm,row.head_silhouette_y_cm),(row.head_reflection_x_cm,row.head_reflection_y_cm)):
            if np.isfinite([x,y]).all(): pts.append((x/args.arena_width_cm*(rw-1),y/args.arena_height_cm*(rh-1)))
        radius=max([12]+[np.hypot(x-cx,y-cy) for x,y in pts]) + 11
        ellipse=cv2.ellipse2Poly((int(cx),int(cy)),(int(radius),int(max(9,radius*.65))),0,0,360,20).astype(np.float32)
        return simplify(cv2.perspectiveTransform(ellipse[None],inverse)[0], 8)
    def load():
        frame=frames[state['i']]; row=labels.get(frame)
        if row is not None and isinstance(row.polygon_px,str) and row.polygon_px:
            state['points']=normalize_polygon(row.polygon_px).astype(float)
        else: state['points']=auto_polygon(frame)
        state['drag']=None; state['dirty']=False
    def commit_current():
        """Keep the current editable polygon in memory before any navigation."""
        if args.only_modified and not state['dirty']:
            return False
        frame = frames[state['i']]
        previous = labels.get(frame)
        excluded = as_bool(previous.exclude) if previous is not None else False
        labels[frame] = pd.Series({
            'frame': frame,
            'polygon_px': json.dumps(np.asarray(state['points']).round(1).tolist()),
            'exclude': excluded,
            'source': 'incremental_small_mask_correction' if args.only_modified else 'torso_constraint',
        })
        state['touched'].add(frame); state['dirty']=False
        propagate(frame, np.asarray(state['points']).copy())
        return True
    def propagate(anchor_frame, polygon):
        if not args.only_modified or args.propagate_similarity <= 0:
            return
        neighbors=similar_candidate_neighbors(
            cap,anchor_frame,set(frames),args.propagate_similarity,args.propagate_window)
        source=read_frame(cap,anchor_frame)
        if source is None: return
        added=0
        for target_frame,similarity in neighbors:
            if target_frame in state['touched']: continue
            # Never overwrite an existing human label. Propagated rows may be
            # replaced when a later, better manual anchor reaches them.
            existing=labels.get(target_frame)
            if existing is not None and 'similarity_propagated' not in str(existing.get('source','')):
                continue
            target=read_frame(cap,target_frame)
            if target is None: continue
            moved=translate_polygon_optical_flow(source,target,polygon)
            if moved is None: continue
            labels[target_frame]=pd.Series({
                'frame':target_frame,
                'polygon_px':json.dumps(moved.round(1).tolist()),
                'exclude':False,
                'source':f'incremental_similarity_propagated_{similarity:.3f}'})
            state['touched'].add(target_frame); added+=1
        state['propagated']+=added
        if added:
            print(f'Frame {anchor_frame}: propagated polygon correction to {added} similar frames')
    def nearest(x,y):
        distance=np.linalg.norm(state['points']-np.asarray([x,y]),axis=1)
        return int(distance.argmin()),float(distance.min())
    def mouse(event,x,y,flags,param):
        pts=state['points']
        if event==cv2.EVENT_LBUTTONDOWN:
            idx,dist=nearest(x,y)
            if dist<16: state['drag']=idx
            else:
                # Insert at nearest edge, preserving a closed contour.
                edges=np.roll(pts,-1,axis=0)-pts; q=np.asarray([x,y])-pts
                t=np.clip(np.sum(q*edges,axis=1)/np.maximum(np.sum(edges*edges,axis=1),1e-6),0,1)
                d=np.linalg.norm(q-edges*t[:,None],axis=1); idx=int(d.argmin())+1
                state['points']=np.insert(pts,idx,[x,y],axis=0); state['drag']=idx
            state['dirty']=True
        elif event==cv2.EVENT_MOUSEMOVE and state['drag'] is not None: state['points'][state['drag']]=[x,y]
        elif event==cv2.EVENT_LBUTTONUP: state['drag']=None
    def save():
        commit_current()
        rows=[]
        selected=(state['touched'] if args.only_modified else labels.keys())
        for frame in selected:
            row=labels[frame]
            rows.append({'frame':frame,'polygon_px':row.polygon_px,
                         'exclude':as_bool(row.exclude),'video':Path(args.input).name,
                         'source':row.get('source','incremental_small_mask_correction' if args.only_modified else 'torso_constraint')})
        if not rows:
            print('No polygon was modified; no training label was added.')
            return
        # Merge into the shared CSV: this session only replaces its own
        # (video, frame) rows; every other video's annotations are kept.
        merge_labels(Path(args.output), Path(args.input).name, rows)
        merge_annotation_exclusions(args.candidate_csv, Path(args.input).name, rows)
    cv2.namedWindow(title,cv2.WINDOW_NORMAL); cv2.setMouseCallback(title,mouse); load()
    while True:
        # Closing the window with X must SAVE, not discard: otherwise the
        # whole session's labels exist only in memory and reappear as
        # "unlabelled" on the next run (the classic re-annotate bug).
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            save(); print('Window closed - annotations saved.')
            break
        frame=frames[state['i']]; img=frame_image(frame); pts=state['points'].astype(np.int32)
        cv2.polylines(img,[pts],True,(255,255,0),2,cv2.LINE_AA)
        for x,y in pts: cv2.circle(img,(x,y),4,(255,255,0),-1,cv2.LINE_AA)
        cv2.rectangle(img,(0,0),(img.shape[1],47),(0,0,0),-1)
        excluded_flag = bool(labels.get(frame) is not None and as_bool(labels[frame].exclude))
        status = ' [EXCLUDED]' if excluded_flag else ''
        cv2.putText(img,f'{state["i"]+1}/{len(frames)} frame={frame}{status}: automatic body contour; propagated={state["propagated"]}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,255) if excluded_flag else (255,255,255),1)
        cv2.putText(img,'Drag vertex / click edge add  Backspace delete  R recalc  T thin(8pts)  E discard  A/D step  S save  Q quit',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1)
        cv2.imshow(title,img); k=cv2.waitKey(20)&255
        if k in (27,ord('q')): save(); break
        if k==ord('s'):
            save()
            if state['i'] < len(frames)-1:
                state['i'] += 1; load()
        elif k==ord('r'): state['points']=auto_polygon(frame); state['dirty']=True
        elif k==ord('t'): state['points']=simplify(np.asarray(state['points']),8); state['dirty']=True
        elif k in (8,127) and len(state['points'])>3:
            idx,_=nearest(*state['points'].mean(axis=0)); state['points']=np.delete(state['points'],idx,axis=0); state['dirty']=True
        elif k in (ord('e'),ord('E')):
            # Toggle exclusion so an accidentally discarded old label can be restored.
            commit_current()
            excluded = not as_bool(labels[frame].exclude)
            labels[frame] = pd.Series({'frame':frame,'polygon_px':json.dumps(np.asarray(state['points']).round(1).tolist()),'exclude':excluded})
            print(f'Frame {frame}: exclude={excluded}')
            state['i'] = min(len(frames)-1, state['i']+1); load()
        elif k in (81,ord('a')): commit_current(); state['i']=max(0,state['i']-1); load()
        elif k in (83,ord('d')): commit_current(); state['i']=min(len(frames)-1,state['i']+1); load()
        elif k==ord('j'): commit_current(); state['i']=max(0,state['i']-10); load()
        elif k==ord('l'): commit_current(); state['i']=min(len(frames)-1,state['i']+10); load()
    cap.release()
    if mask_cap is not None: mask_cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__': main()
