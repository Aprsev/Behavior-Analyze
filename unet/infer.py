#!/usr/bin/env python3
"""Run U-Net segmentation; export mask/overlay videos and CNN-centroid CSV.

Pass --exclude-csv (the screening CSV from unet/screen_frames.py) to make
manually excluded frames visible in the outputs: their trajectory rows become
NaN, the mask video shows a black frame, and the overlay video marks them with
a red EXCLUDED label. The excluded frames are also reported in inference.json.

Pass --rotate 90/180/270 when the arena was turned by a quarter-turn compared
to the training videos: frames are rotated before the CNN and the mask /
centroid are rotated back, so the model always sees the training orientation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from model import UNet

def rot_pt(p,k,h,w):
 """(col,row) in an (h,w) frame -> (col,row) in np.rot90(frame,k)."""
 x,y=p
 for _ in range(k%4):
  x,y=y,w-1-x;h,w=w,h
 return x,y

def inv_rot_pt(p,k,h,w):
 """inverse of rot_pt (p is in the rotated frame)."""
 x,y=p
 for _ in range(k%4):h,w=w,h
 for _ in range(k%4):
  x,y=h-1-y,x;h,w=w,h
 return x,y

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--model',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--rotate',type=int,default=0,choices=[0,90,180,270]);p.add_argument('--exclude-csv',default='');a=p.parse_args()
 excluded=set()
 if a.exclude_csv and Path(a.exclude_csv).is_file():
  ex=pd.read_csv(a.exclude_csv);ex=ex.loc[ex.exclude.fillna(False).astype(bool)&(ex.video==Path(a.video).name)]
  excluded=set(int(f) for f in ex.frame);print(f'Marking {len(excluded)} screened frames as excluded')
 pack=torch.load(a.model,map_location='cpu');size=int(pack['size']);dev='cuda' if torch.cuda.is_available() else 'cpu';net=UNet().to(dev);net.load_state_dict(pack['state_dict']);net.eval();cap=cv2.VideoCapture(a.video);fps=cap.get(cv2.CAP_PROP_FPS);w,h=int(cap.get(3)),int(cap.get(4));k=int(a.rotate)//90%4
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);fourcc=cv2.VideoWriter_fourcc(*'mp4v');mw=cv2.VideoWriter(str(out/'mouse_miniscope_mask.mp4'),fourcc,fps,(w,h));ow=cv2.VideoWriter(str(out/'mouse_miniscope_overlay.mp4'),fourcc,fps,(w,h));rows=[];i=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  if i in excluded:
   clean=np.zeros((h,w),np.uint8);cx=cy=float('nan');overlay=frame.copy()
   cv2.rectangle(overlay,(0,0),(w-1,h-1),(0,0,220),6);cv2.putText(overlay,f'EXCLUDED frame {i}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,220),2)
  else:
   frame_in=np.rot90(frame,k) if k else frame;hi,wi=frame_in.shape[:2]
   gray=cv2.cvtColor(frame_in,cv2.COLOR_BGR2GRAY);small=cv2.resize(gray,(size,size));x=torch.from_numpy(small[None,None].copy()).float().to(dev)/255
   with torch.no_grad():prob=torch.sigmoid(net(x))[0,0].cpu().numpy()
   mask=(cv2.resize(prob,(wi,hi))>=a.threshold).astype(np.uint8)*255;n,labels,stats,cent=cv2.connectedComponentsWithStats(mask)
   if n>1:
    label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));clean=(labels==label).astype(np.uint8)*255;cx,cy=cent[label]
    if k:cx,cy=inv_rot_pt((float(cx),float(cy)),k,h,w)
   else:clean=np.zeros_like(mask);cx=cy=np.nan
   if k:clean=np.rot90(clean,-k)
   overlay=frame.copy();overlay[clean>0]=(0,220,0);overlay=cv2.addWeighted(frame,.65,overlay,.35,0)
  mw.write(cv2.cvtColor(clean,cv2.COLOR_GRAY2BGR));ow.write(overlay);rows.append((i,i/fps,cx,cy));i+=1
 cap.release();mw.release();ow.release();pd.DataFrame(rows,columns=['frame','timestamp_sec','body_x_px','body_y_px']).to_csv(out/'unet_trajectory.csv',index=False)
 manifest={'device':dev,'frames':i,'threshold':a.threshold,'rotate':a.rotate,'excluded_frames':sorted(excluded&set(range(i))),'excluded_count':len(excluded&set(range(i)))}
 (out/'inference.json').write_text(json.dumps(manifest,indent=2));print(f'Wrote {i} frames to {out} (excluded {manifest["excluded_count"]})')
if __name__=='__main__':main()