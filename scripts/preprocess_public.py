#!/usr/bin/env python
from pathlib import Path
import argparse,json,sys,yaml,numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tidalnet.data.io import save_npz


def load_manifest(path: str):
    obj=json.loads(Path(path).read_text()); channels=obj.get("channels",[])
    if not channels: raise SystemExit(f"No channels listed in {path}")
    return channels,obj.get("status","UNKNOWN")


def locate_channel(root: Path, split: str, channel: str) -> Path:
    candidates=[root/"data"/split/f"{channel}.npy",root/split/f"{channel}.npy",root/f"{channel}.npy"]
    for p in candidates:
        if p.exists(): return p
    matches=list(root.rglob(f"{channel}.npy"))
    matches=[p for p in matches if split.lower() in str(p.parent).lower()] or matches
    if not matches: raise FileNotFoundError(f"Cannot find {split} array for channel {channel} below {root}")
    return matches[0]


def preprocess_nasa(cfg: dict, raw_root: Path, split: str, max_samples: int|None):
    channels,status=load_manifest(cfg["channel_manifest"]); series=[]
    for ch in channels:
        arr=np.load(locate_channel(raw_root,split,ch))
        if arr.ndim==1: y=arr
        elif arr.ndim==2: y=arr[:,0]  # Telemanom first column is the telemetry target.
        else: raise ValueError(f"Unexpected shape {arr.shape} for {ch}")
        series.append(np.asarray(y,dtype=np.float32))
    length=min(map(len,series)); length=min(length,max_samples) if max_samples else length
    values=np.stack([x[:length] for x in series],axis=1)
    natural=np.isfinite(values).astype(np.float32); values=np.nan_to_num(values,nan=0.0)
    times=np.arange(length,dtype=np.float64)*float(cfg["sample_interval_seconds"])
    save_npz(cfg["output"],values=values,reference_times=times,natural_mask=natural,
             channel_names=np.asarray(channels),manifest_status=np.asarray(status),
             preprocessing_json=np.asarray(json.dumps({"source":"Telemanom","split":split,"alignment":"truncate_to_minimum_length","selected_column":0,"max_samples":max_samples})))

p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=["msl","smap","esa_adb"],required=True); p.add_argument("--config",required=True); p.add_argument("--raw-root"); p.add_argument("--split",default="train"); p.add_argument("--max-samples",type=int,default=20000); p.add_argument("--matrix",help="Aligned matrix fallback for ESA or author-preprocessed release"); a=p.parse_args()
cfg=yaml.safe_load(Path(a.config).read_text())["dataset"]
if a.dataset in {"msl","smap"}:
    raw=Path(a.raw_root or "data/raw/telemanom"); preprocess_nasa(cfg,raw,a.split,a.max_samples)
else:
    if not a.matrix: raise SystemExit("ESA-ADB layouts differ by release. Supply the exact aligned --matrix used by the paper and pin its checksum in the manifest.")
    path=Path(a.matrix); z=np.load(path) if path.suffix in {".npy",".npz"} else None
    if path.suffix==".npy": values=z
    elif path.suffix==".npz": values=z["values"] if "values" in z else z[z.files[0]]
    else:
        import pandas as pd; values=pd.read_csv(path).to_numpy()
    values=np.asarray(values,dtype=np.float32); natural=np.isfinite(values).astype(np.float32); values=np.nan_to_num(values,nan=0.0)
    times=np.arange(len(values),dtype=np.float64)*float(cfg["sample_interval_seconds"])
    save_npz(cfg["output"],values=values,reference_times=times,natural_mask=natural)
print(cfg["output"])
