#!/usr/bin/env python3
"""Run U-Net segmentation; export mask/overlay videos and CNN-centroid CSV."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from model import UNet
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--model',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--threshold',type=float,default=.5);a=p.parse_args();pack=torch.load(a.model,map_location='cpu');size=int(pack['size']);dev='cuda' if torch.cuda.is_available() else 'cpu';net=UNet().to(dev);net.load_state_dict(pack['state_dict']);net.eval();cap=cv2.VideoCapture(a.video);fps=cap.get(cv2.CAP_PROP_FPS);w,h=int(cap.get(3)),int(cap.get(4));out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);fourcc=cv2.VideoWriter_fourcc(*'mp4v');mw=cv2.VideoWriter(str(out/'mouse_miniscope_mask.mp4'),fourcc,fps,(w,h));ow=cv2.VideoWriter(str(out/'mouse_miniscope_overlay.mp4'),fourcc,fps,(w,h));rows=[];i=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);small=cv2.resize(gray,(size,size));x=torch.from_numpy(small[None,None].copy()).float().to(dev)/255
  with torch.no_grad():prob=torch.sigmoid(net(x))[0,0].cpu().numpy()
  mask=(cv2.resize(prob,(w,h))>=a.threshold).astype(np.uint8)*255;n,labels,stats,cent=cv2.connectedComponentsWithStats(mask)
  if n>1:
   label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));clean=(labels==label).astype(np.uint8)*255;cx,cy=cent[label]
  else:clean=np.zeros_like(mask);cx=cy=np.nan
  overlay=frame.copy();overlay[clean>0]=(0,220,0);overlay=cv2.addWeighted(frame,.65,overlay,.35,0);mw.write(cv2.cvtColor(clean,cv2.COLOR_GRAY2BGR));ow.write(overlay);rows.append((i,i/fps,cx,cy));i+=1
 cap.release();mw.release();ow.release();pd.DataFrame(rows,columns=['frame','timestamp_sec','body_x_px','body_y_px']).to_csv(out/'unet_trajectory.csv',index=False);(out/'inference.json').write_text(json.dumps({'device':dev,'frames':i,'threshold':a.threshold},indent=2));print(f'Wrote {i} frames to {out}')
if __name__=='__main__':main()
