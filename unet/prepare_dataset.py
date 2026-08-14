#!/usr/bin/env python3
"""Convert saved irregular torso polygons into images/masks for U-Net."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--labels',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args()
 labels=pd.read_csv(a.labels); labels=labels.loc[~labels.exclude.fillna(False).astype(bool)]
 if 'polygon_px' not in labels: raise ValueError('labels must be polygon-based manual_torso_constraints.csv')
 out=Path(a.output_dir); images=out/'images'; masks=out/'masks';images.mkdir(parents=True,exist_ok=True);masks.mkdir(parents=True,exist_ok=True)
 cap=cv2.VideoCapture(a.video);written=[]
 for _,row in labels.iterrows():
  if not isinstance(row.polygon_px,str) or not row.polygon_px:continue
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(row.frame));ok,frame=cap.read()
  if not ok:continue
  polygon=np.asarray(json.loads(row.polygon_px),np.float32); mask=np.zeros(frame.shape[:2],np.uint8);cv2.fillPoly(mask,[polygon.astype(np.int32)],255)
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);gray=cv2.resize(gray,(a.size,a.size),interpolation=cv2.INTER_AREA);mask=cv2.resize(mask,(a.size,a.size),interpolation=cv2.INTER_NEAREST)
  name=f'{int(row.frame):07d}.png';cv2.imwrite(str(images/name),gray);cv2.imwrite(str(masks/name),mask);written.append(int(row.frame))
 cap.release();(out/'dataset.json').write_text(json.dumps({'video':str(Path(a.video).resolve()),'count':len(written),'size':a.size,'frames':written},indent=2),encoding='utf-8');print(f'Wrote {len(written)} image/mask pairs to {out}')
if __name__=='__main__':main()
