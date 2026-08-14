"""Small U-Net for binary mouse-plus-miniscope segmentation."""
from __future__ import annotations
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, base: int=24):
        super().__init__()
        self.e1=Block(1,base); self.e2=Block(base,base*2); self.e3=Block(base*2,base*4)
        self.pool=nn.MaxPool2d(2); self.mid=Block(base*4,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=Block(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=Block(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=Block(base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); m=self.mid(self.pool(c))
        x=self.d3(torch.cat([self.u3(m),c],1)); x=self.d2(torch.cat([self.u2(x),b],1)); x=self.d1(torch.cat([self.u1(x),a],1)); return self.out(x)
