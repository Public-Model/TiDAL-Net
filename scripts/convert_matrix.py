#!/usr/bin/env python
from pathlib import Path
import argparse,sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tidalnet.data.io import save_npz
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--sample-interval",type=float,default=60.0); p.add_argument("--has-header",action="store_true"); a=p.parse_args()
path=Path(a.input)
if path.suffix==".npy": values=np.load(path)
elif path.suffix==".npz":
    z=np.load(path); values=z["values"] if "values" in z else z[z.files[0]]
else: values=pd.read_csv(path,header=0 if a.has_header else None).to_numpy()
values=np.asarray(values,dtype=np.float32)
if values.ndim!=2: raise SystemExit("input matrix must have shape [T, M]")
natural=np.isfinite(values).astype(np.float32); values=np.nan_to_num(values,nan=0.0); times=np.arange(len(values),dtype=np.float64)*a.sample_interval
save_npz(a.output,values=values,reference_times=times,natural_mask=natural); print(a.output)
