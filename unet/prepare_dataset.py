#!/usr/bin/env python3
"""Convert saved irregular torso polygons into images/masks for U-Net.

Supports multiple videos: run once per video with the same --output-dir and
--labels; image names are prefixed with the video stem so frames never
collide, and dataset.json accumulates one entry per video. Pass
--exclude-csv (the screening CSV from unet/screen_frames.py) to drop frames
manually flagged during screening (mouse absent, human intervention, ...).
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--labels',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--size',type=int,default=256);p.add_argument('--exclude-csv',default='');a=p.parse_args()
 labels=pd.read_csv(a.labels); labels=labels.loc[~labels.exclude.fillna(False).astype(bool)]
 if 'polygon_px' not in labels: raise ValueError('labels must be polygon-based manual_torso_constraints.csv')
 if 'video' in labels.columns:
  # Multi-video labels: each row belongs to one recording; the same frame
  # number in another video is a different image, so filter strictly.
  labels=labels.loc[labels.video==Path(a.video).name]
  print(f'Using {len(labels)} labels for {Path(a.video).name}')
 excluded=set()
 if a.exclude_csv and Path(a.exclude_csv).is_file():
  ex=pd.read_csv(a.exclude_csv);ex=ex.loc[ex.exclude.fillna(False).astype(bool)&(ex.video==Path(a.video).name)]
  excluded=set(int(f) for f in ex.frame);print(f'Excluding {len(excluded)} screened frames of {Path(a.video).name}')
 out=Path(a.output_dir); images=out/'images'; masks=out/'masks';images.mkdir(parents=True,exist_ok=True);masks.mkdir(parents=True,exist_ok=True)
 stem=Path(a.video).stem.replace(' ','_');cap=cv2.VideoCapture(a.video);written=[]
 for _,row in labels.iterrows():
  frame_index=int(row.frame)
  if frame_index in excluded or not isinstance(row.polygon_px,str) or not row.polygon_px:continue
  cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,frame=cap.read()
  if not ok:continue
  polygon=np.asarray(json.loads(row.polygon_px),np.float32); mask=np.zeros(frame.shape[:2],np.uint8);cv2.fillPoly(mask,[polygon.astype(np.int32)],255)
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);gray=cv2.resize(gray,(a.size,a.size),interpolation=cv2.INTER_AREA);mask=cv2.resize(mask,(a.size,a.size),interpolation=cv2.INTER_NEAREST)
  name=f'{stem}_{frame_index:07d}.png';cv2.imwrite(str(images/name),gray);cv2.imwrite(str(masks/name),mask);written.append(frame_index)
 cap.release()
 manifest={'video':str(Path(a.video).resolve()),'count':len(written),'size':a.size,'frames':written}
 if (out/'dataset.json').exists():
  old=json.loads((out/'dataset.json').read_text(encoding='utf-8'))
  videos=old.get('videos') or []
  videos=[v for v in videos if v.get('video')!=manifest['video']];videos.append(manifest)
 else: videos=[manifest]
 (out/'dataset.json').write_text(json.dumps({'size':a.size,'videos':videos},indent=2),encoding='utf-8')
 print(f'Wrote {len(written)} image/mask pairs from {Path(a.video).name} to {out} (total videos: {len(videos)})')
if __name__=='__main__':main()
