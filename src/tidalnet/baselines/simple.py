from __future__ import annotations
import numpy as np
import torch
from torch import nn


def persistence_fill(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out=values.copy()
    for ch in range(out.shape[1]):
        last=0.0
        for t in range(out.shape[0]):
            if mask[t,ch]>0: last=out[t,ch]
            else: out[t,ch]=last
    return out


def linear_interpolation(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out=values.copy(); idx=np.arange(len(values))
    for ch in range(out.shape[1]):
        valid=mask[:,ch]>0
        if valid.sum()>=2: out[:,ch]=np.interp(idx,idx[valid],values[valid,ch])
        elif valid.sum()==1: out[:,ch]=values[valid,ch][0]
    return out


class GRUReconstructor(nn.Module):
    def __init__(self, channels: int, hidden: int=64):
        super().__init__(); self.gru=nn.GRU(channels*2,hidden,batch_first=True); self.out=nn.Linear(hidden,channels)
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h,_=self.gru(torch.cat([values,mask],-1)); return self.out(h)
