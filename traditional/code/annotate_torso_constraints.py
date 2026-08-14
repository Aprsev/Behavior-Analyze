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
vertex; R restores automatic contour; E excludes frame; A/D step; J/L +/-10;
S save; Q/Esc save and quit.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from mouse_behavior_pipeline import (
    perspective_geometry, robust_threshold, sample_frames, segment_mouse, video_properties,
)


def simplify(points: np.ndarray, count: int = 16) -> np.ndarray:
    if len(points) <= count: return points.astype(float)
    return points[np.linspace(0, len(points)-1, count, dtype=int)].astype(float)


def scan_candidates(
    cap: cv2.VideoCapture, total: int, forward: np.ndarray, rw: int, rh: int,
    background: np.ndarray, threshold: float, pool_cap: int = 2000,
    excluded: set = frozenset(),
) -> tuple[list, list]:
    """Scan a uniform grid of frames; return (good, junk) candidate lists.

    good: (frame, normalized 48x48 arena descriptor, rectified torso contour)
    junk: (frame, None) where segmentation failed or mask size is implausible
    (mouse absent, human intervention, severe occlusion, motion blur, ...).
    """
    step = max(1, total // pool_cap)
    good, junk = [], []
    for f in range(0, total, step):
        if f in excluded:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, frame = cap.read()
        if not ok:
            continue
        rect = cv2.warpPerspective(frame, forward, (rw, rh))
        detection = segment_mouse(rect, background, threshold, None, float('inf'), None)
        if detection.contour is None or not (0.001 * rect.size <= detection.area <= 0.2 * rect.size):
            junk.append((f, None))
            continue
        gray = cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (48, 48)).astype(np.float32)
        gray = (gray - gray.mean()) / (gray.std() + 1e-6)
        good.append((f, gray.ravel(), detection.contour))
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
        cand = cand.loc[cand.video == Path(args.input).name]
        if not len(cand):
            raise SystemExit(f'No screening rows for {Path(args.input).name}.\n'
                             f'Run screening first: python unet/run_unet.py screen')
        keep = cand.loc[~cand.exclude.fillna(False).astype(bool)].frame.astype(int)
        frames = [f for f in dict.fromkeys(int(f) for f in keep) if f not in labels]
        print(f'Using {len(frames)} screened candidate frames (junk already removed).')
        return frames[:args.max_labels]
    picks = pick_diverse_frames(cap, total, forward, rw, rh, background, threshold, args.max_labels, set())
    return [f for f, _, _ in picks if f not in labels]


def merge_annotation_exclusions(screening_csv: str, video_name: str, label_rows: list[dict]) -> None:
    """Record frames excluded during annotation (E key) in the screening CSV,
    so prepare/infer also skip them. Silently skips when no screening CSV."""
    if not screening_csv or not Path(screening_csv).is_file():
        return
    excluded = {int(row['frame']) for row in label_rows if bool(row.get('exclude'))}
    if not excluded:
        return
    path = Path(screening_csv)
    df = pd.read_csv(path)
    df.loc[(df.video == video_name) & df.frame.isin(excluded), 'exclude'] = 1
    df.to_csv(path, index=False)
    print(f'Recorded {len(excluded)} annotation exclusions in {screening_csv}')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--comparison-csv',required=True)
    p.add_argument('--roi-json',required=True); p.add_argument('--output',required=True)
    p.add_argument('--arena-width-cm',type=float,required=True); p.add_argument('--arena-height-cm',type=float,required=True)
    p.add_argument('--max-labels',type=int,default=50); p.add_argument('--candidate-csv',default='')
    args=p.parse_args()
    data=pd.read_csv(args.comparison_csv)
    old=pd.read_csv(args.output) if Path(args.output).exists() else pd.DataFrame(columns=['frame','polygon_px','exclude'])
    labels={int(row.frame):row for _,row in old.iterrows()}
    corners=np.asarray(json.loads(Path(args.roi_json).read_text(encoding='utf-8'))['arena_corners_px'],np.float32)
    total,fps,_,_=video_properties(Path(args.input))
    rw,rh,forward,inverse,_,_=perspective_geometry(corners,args.arena_width_cm,args.arena_height_cm)
    _,samples=sample_frames(Path(args.input),total,61)
    rect_samples=np.stack([cv2.warpPerspective(frame,forward,(rw,rh)) for frame in samples])
    background=np.percentile(rect_samples,85,axis=0).astype(np.uint8)
    threshold,_=robust_threshold(rect_samples,background,0)
    cap=cv2.VideoCapture(args.input); title='Polygon torso correction (cyan); remove fibre/tail'
    frames=_select_frames(args,labels,cap,total,forward,rw,rh,background,threshold)
    if not frames:
        print('No new frames to label: all screened candidates are already labelled.')
        cap.release(); return
    state={'i':0,'points':None,'drag':None}
    def frame_image(frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,img=cap.read()
        if not ok: raise RuntimeError(f'Cannot read frame {frame}')
        return img
    def auto_polygon(frame):
        # The same dark-foreground / compact-torso segmentation used in the
        # tracking pipeline supplies the initial editable contour.
        image=frame_image(frame)
        rectified=cv2.warpPerspective(image,forward,(rw,rh))
        detection=segment_mouse(rectified,background,threshold,None,float('inf'),None)
        if detection.contour is not None:
            source_contour=cv2.perspectiveTransform(detection.contour.astype(np.float32),inverse)[:,0,:]
            return simplify(source_contour,16)
        row=data.loc[data.frame==frame].iloc[0]
        # Fallback only when automatic segmentation failed entirely.
        cx,cy=row.body_x_cm/args.arena_width_cm*(rw-1),row.body_y_cm/args.arena_height_cm*(rh-1)
        pts=[]
        for x,y in ((row.head_silhouette_x_cm,row.head_silhouette_y_cm),(row.head_reflection_x_cm,row.head_reflection_y_cm)):
            if np.isfinite([x,y]).all(): pts.append((x/args.arena_width_cm*(rw-1),y/args.arena_height_cm*(rh-1)))
        radius=max([12]+[np.hypot(x-cx,y-cy) for x,y in pts]) + 11
        ellipse=cv2.ellipse2Poly((int(cx),int(cy)),(int(radius),int(max(9,radius*.65))),0,0,360,20).astype(np.float32)
        return cv2.perspectiveTransform(ellipse[None],inverse)[0]
    def load():
        frame=frames[state['i']]; row=labels.get(frame)
        if row is not None and isinstance(row.polygon_px,str) and row.polygon_px:
            state['points']=np.asarray(json.loads(row.polygon_px),float)
        else: state['points']=auto_polygon(frame)
        state['drag']=None
    def commit_current():
        """Keep the current editable polygon in memory before any navigation."""
        frame = frames[state['i']]
        previous = labels.get(frame)
        excluded = bool(previous.exclude) if previous is not None else False
        labels[frame] = pd.Series({
            'frame': frame,
            'polygon_px': json.dumps(np.asarray(state['points']).round(1).tolist()),
            'exclude': excluded,
        })
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
        elif event==cv2.EVENT_MOUSEMOVE and state['drag'] is not None: state['points'][state['drag']]=[x,y]
        elif event==cv2.EVENT_LBUTTONUP: state['drag']=None
    def save():
        commit_current()
        rows=[]
        for frame,row in labels.items(): rows.append({'frame':frame,'polygon_px':row.polygon_px,'exclude':bool(row.exclude)})
        Path(args.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).sort_values('frame').to_csv(args.output,index=False)
        print(f'Saved {len(rows)} polygon torso constraints to {args.output}')
        merge_annotation_exclusions(args.candidate_csv, Path(args.input).name, rows)
    cv2.namedWindow(title,cv2.WINDOW_NORMAL); cv2.setMouseCallback(title,mouse); load()
    while True:
        frame=frames[state['i']]; img=frame_image(frame); pts=state['points'].astype(np.int32)
        cv2.polylines(img,[pts],True,(255,255,0),2,cv2.LINE_AA)
        for x,y in pts: cv2.circle(img,(x,y),4,(255,255,0),-1,cv2.LINE_AA)
        cv2.rectangle(img,(0,0),(img.shape[1],47),(0,0,0),-1)
        cv2.putText(img,f'{state["i"]+1}/{len(frames)} frame={frame}: automatic body contour; adjust only fibre/tail-affected vertices',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        cv2.putText(img,'Drag vertex / click edge add  Backspace delete  R recalculate auto  A/D step  S save  Q quit',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1)
        cv2.imshow(title,img); k=cv2.waitKey(20)&255
        if k in (27,ord('q')): save(); break
        if k==ord('s'):
            save()
            if state['i'] < len(frames)-1:
                state['i'] += 1; load()
        elif k==ord('r'): state['points']=auto_polygon(frame)
        elif k in (8,127) and len(state['points'])>3:
            idx,_=nearest(*state['points'].mean(axis=0)); state['points']=np.delete(state['points'],idx,axis=0)
        elif k in (81,ord('a')): commit_current(); state['i']=max(0,state['i']-1); load()
        elif k in (83,ord('d')): commit_current(); state['i']=min(len(frames)-1,state['i']+1); load()
        elif k==ord('j'): commit_current(); state['i']=max(0,state['i']-10); load()
        elif k==ord('l'): commit_current(); state['i']=min(len(frames)-1,state['i']+10); load()
    cap.release();cv2.destroyAllWindows()

if __name__=='__main__': main()
