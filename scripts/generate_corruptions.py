#!/usr/bin/env python
from pathlib import Path
import argparse,sys,numpy as np,yaml,json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tidalnet.data.io import load_npz,save_npz
from tidalnet.data.corruption import CorruptionConfig,apply_corruption
from tidalnet.data.windows import chronological_bounds
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--graph"); p.add_argument("--config",default="configs/corruption/reality_informed.yaml"); p.add_argument("--ratio",type=float); p.add_argument("--seed",type=int,default=2026); p.add_argument("--global-mask",action="store_true",help="Do not generate each chronological split independently"); a=p.parse_args()
raw=yaml.safe_load(Path(a.config).read_text()); raw["short_burst_length"]=tuple(raw["short_burst_length"]); raw["long_gap_length"]=tuple(raw["long_gap_length"]); raw["correlated_group_size"]=tuple(raw["correlated_group_size"])
if a.ratio is not None: raw["affected_ratio"]=a.ratio
cfg=CorruptionConfig(**raw); d=load_npz(a.input); graph=np.load(a.graph) if a.graph else None
if a.global_mask:
    out=apply_corruption(d["values"],d["reference_times"],d["natural_mask"],cfg,a.seed,graph)
else:
    bounds=chronological_bounds(len(d["values"]),(0.6,0.2,0.2)); chunks=[]; split_meta={}
    for idx,(name,(start,end)) in enumerate(bounds.items()):
        chunk=apply_corruption(d["values"][start:end],d["reference_times"][start:end],d["natural_mask"][start:end],cfg,a.seed+idx,graph)
        split_meta[name]=json.loads(str(chunk.pop("metadata_json")))
        chunks.append(chunk)
    out={key:np.concatenate([c[key] for c in chunks],axis=0) for key in chunks[0]}
    out["metadata_json"]=np.asarray(json.dumps({"mode":"split_wise","base_seed":a.seed,"splits":split_meta}))
save_npz(a.output,**out); print(a.output)
