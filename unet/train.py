#!/usr/bin/env python3
"""Train small U-Net on polygon masks; selects best validation Dice checkpoint."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import DataLoader,Dataset
from model import UNet

class Pairs(Dataset):
 def __init__(self,root,files,augment=False):self.root=Path(root);self.files=files;self.augment=augment
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  f=self.files[i];x=cv2.imread(str(self.root/'images'/f),0);y=cv2.imread(str(self.root/'masks'/f),0)
  if self.augment:
   # 90-degree family: covers arena rotations (e.g. camera/box turned 90,
   # 180 or 270 degrees) with zero border artifacts - np.rot90 has none.
   k=random.randint(0,3)
   if k:x,y=np.rot90(x,k).copy(),np.rot90(y,k).copy()
   if random.random()<.5:x,y=np.fliplr(x).copy(),np.fliplr(y).copy()
   if random.random()<.5:x,y=np.flipud(x).copy(),np.flipud(y).copy()
   angle=random.uniform(-18,18);m=cv2.getRotationMatrix2D((x.shape[1]/2,x.shape[0]/2),angle,1);x=cv2.warpAffine(x,m,x.shape[::-1],borderMode=cv2.BORDER_REFLECT);y=cv2.warpAffine(y,m,y.shape[::-1],flags=cv2.INTER_NEAREST)
   # lighting: affine brightness + non-linear gamma curve (different lamps,
   # white balance) - gamma is what linear scale+offset cannot model
   x=np.clip(x.astype(np.float32)*random.uniform(.75,1.25)+random.uniform(-12,12),0,255)
   if random.random()<.6:x=255.0*(x/255.0)**random.uniform(.7,1.4)
   # sensor noise + mild blur (different camera / focus)
   if random.random()<.5:x+=np.random.normal(0,random.uniform(1,6),x.shape)
   if random.random()<.3:x=cv2.GaussianBlur(np.clip(x,0,255).astype(np.uint8),(3,3),0)
   x=np.clip(x,0,255).astype(np.uint8)
  return torch.from_numpy(x[None].copy()).float()/255,torch.from_numpy((y[None]>127).copy()).float()
def dice(logits,target):
 p=torch.sigmoid(logits);return (2*(p*target).sum()+1)/((p+target).sum()+1)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--epochs',type=int,default=80);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=2e-3);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);root=Path(a.dataset);files=sorted(x.name for x in (root/'images').glob('*.png'))
 if len(files)<20:raise ValueError(f'Need >=20 masks, found {len(files)}')
 random.shuffle(files);n=max(1,round(len(files)*.2));train,val=files[n:],files[:n];dev='cuda' if torch.cuda.is_available() else 'cpu';model=UNet().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);lossfn=torch.nn.BCEWithLogitsLoss();tl=DataLoader(Pairs(root,train,True),batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=dev=='cuda');vl=DataLoader(Pairs(root,val),batch_size=a.batch_size,num_workers=2,pin_memory=dev=='cuda')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);best=-1;history=[]
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for x,y in tl:
   x,y=x.to(dev),y.to(dev);z=model(x);loss=lossfn(z,y)+(1-dice(z,y));opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
  model.eval();scores=[]
  with torch.no_grad():
   for x,y in vl:scores += [dice(model(x.to(dev)),y.to(dev)).item()]
  score=float(np.mean(scores));history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_dice':score});print(json.dumps(history[-1]),flush=True)
  if score>best:best=score;torch.save({'state_dict':model.state_dict(),'size':cv2.imread(str(root/'images'/files[0]),0).shape[0],'val_dice':best},out/'best_unet.pt')
 (out/'training_history.json').write_text(json.dumps({'device':dev,'train_count':len(train),'val_count':len(val),'best_val_dice':best,'history':history},indent=2),encoding='utf-8')
if __name__=='__main__':main()
