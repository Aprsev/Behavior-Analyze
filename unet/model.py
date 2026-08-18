"""Backward-compatible multi-task U-Net for mask and keypoint heatmaps."""
from __future__ import annotations
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)


class UNet(nn.Module):
    """U-Net with backward-compatible optional point decoders.

    Old checkpoints used one grayscale channel and only ``out``.  New
    checkpoints use two channels (raw grayscale + background residual) and
    may additionally predict Head and Reflection heatmaps. Keeping the
    architecture in one class lets the loader migrate useful old weights.
    """

    def __init__(self, base: int = 24, in_channels: int = 1,
                 head_output: bool = False, reflection_output: bool = False,
                 reflection_refine: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.reflection_output = bool(reflection_output)
        self.reflection_refine_enabled = bool(reflection_output and reflection_refine)
        self.head_output = bool(head_output or self.reflection_output)
        self.e1=Block(self.in_channels,base); self.e2=Block(base,base*2); self.e3=Block(base*2,base*4)
        self.pool=nn.MaxPool2d(2); self.mid=Block(base*4,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=Block(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=Block(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=Block(base*2,base)
        self.out=nn.Conv2d(base,1,1)
        self.head_out = nn.Conv2d(base, 1, 1) if self.head_output else None
        self.reflection_refine = (nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
            if self.reflection_refine_enabled else None)
        self.reflection_out = nn.Conv2d(base, 1, 1) if self.reflection_output else None
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); m=self.mid(self.pool(c))
        x=self.d3(torch.cat([self.u3(m),c],1)); x=self.d2(torch.cat([self.u2(x),b],1)); x=self.d1(torch.cat([self.u1(x),a],1))
        mask = self.out(x)
        if self.head_out is None:
            return mask
        head = self.head_out(x)
        if self.reflection_out is None:
            return mask, head
        reflection_features = self.reflection_refine(x) if self.reflection_refine is not None else x
        return mask, head, self.reflection_out(reflection_features)


def unpack_outputs(output):
    """Normalize legacy/new forward results to mask, head, reflection."""
    if not isinstance(output, tuple):
        return output, None, None
    if len(output) == 2:
        return output[0], output[1], None
    if len(output) == 3:
        return output
    raise ValueError(f"Unsupported U-Net output count: {len(output)}")


def checkpoint_model(package: dict, device="cpu") -> UNet:
    """Build and load the exact architecture recorded in a checkpoint."""
    in_channels = int(package.get("in_channels", 2 if package.get("dual_channel") else 1))
    head_output = bool(package.get("head_output", False))
    reflection_output = bool(package.get("reflection_output", False))
    reflection_refine = bool(package.get("reflection_refine", False))
    model = UNet(in_channels=in_channels, head_output=head_output,
                 reflection_output=reflection_output,
                 reflection_refine=reflection_refine).to(device)
    model.load_state_dict(package["state_dict"])
    return model


def load_compatible_weights(model: UNet, state_dict: dict,
                            source_channel: int = 0) -> list[str]:
    """Warm-start a new model from an older one-channel checkpoint.

    Matching tensors are copied directly.  For the first convolution, an old
    grayscale kernel is copied into the channel matching the old checkpoint
    (raw or residual) and the other channel starts at zero, so migration
    cannot immediately change the old prediction. Newly introduced head
    layers keep their initialization.
    """
    current = model.state_dict()
    loaded: list[str] = []
    for name, value in state_dict.items():
        if name not in current:
            continue
        target = current[name]
        if value.shape == target.shape:
            current[name] = value
            loaded.append(name)
        elif name == "e1.net.0.weight" and value.ndim == 4 \
                and value.shape[1] == 1 and target.shape[1] == 2 \
                and value.shape[0] == target.shape[0]:
            migrated = target.clone()
            migrated.zero_()
            migrated[:, int(source_channel)] = value[:, 0]
            current[name] = migrated
            loaded.append(name + " (1ch->2ch)")
    model.load_state_dict(current)
    return loaded
