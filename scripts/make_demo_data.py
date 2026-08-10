#!/usr/bin/env python
from pathlib import Path
import argparse, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tidalnet.data.io import save_npz

p=argparse.ArgumentParser(); p.add_argument("--output",default="data/processed/demo.npz"); p.add_argument("--length",type=int,default=160); p.add_argument("--channels",type=int,default=6); p.add_argument("--seed",type=int,default=2026); a=p.parse_args()
rng=np.random.default_rng(a.seed); t=np.arange(a.length,dtype=np.float64)*60.0
base=np.stack([np.sin(np.arange(a.length)/(8+i))+0.3*np.cos(np.arange(a.length)/(17+i)) for i in range(a.channels)],1)
mix=np.eye(a.channels)*0.7+0.3/a.channels; values=base@mix+rng.normal(0,0.03,base.shape)
values=values.astype(np.float32); natural=np.ones_like(values,dtype=np.float32)
save_npz(a.output,values=values,reference_times=t,natural_mask=natural,channel_names=np.asarray([f"demo_{i}" for i in range(a.channels)]))
print(a.output)
