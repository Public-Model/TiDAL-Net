#!/usr/bin/env python
"""Generate one immutable corruption file and resolved configuration per seed, then train/evaluate.

Use this with a single-dataset resolved config containing `data.clean_path`, `data.graph_path`,
and an optional `data.corruption_config`. The same per-seed corrupted NPZ can then be supplied
to every baseline, enabling paired comparisons on identical affected positions.
"""
from pathlib import Path
import argparse,subprocess,sys,yaml,numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tidalnet.config import load_config,save_config
from tidalnet.data.io import load_npz,save_npz
from tidalnet.data.corruption import CorruptionConfig,apply_corruption
from tidalnet.data.windows import chronological_bounds

p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--runs",type=int,default=30); p.add_argument("--ratio",type=float,default=0.05); p.add_argument("--start-seed",type=int,default=2026); p.add_argument("--device"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
cfg=load_config(a.config); clean_path=cfg["data"].get("clean_path")
if not clean_path: raise SystemExit("Repeated runs require data.clean_path pointing to a complete pre-corruption NPZ.")
clean=load_npz(clean_path); graph=np.load(cfg["data"]["graph_path"]); corruption_path=cfg["data"].get("corruption_config","configs/corruption/reality_informed.yaml")
raw=yaml.safe_load(Path(corruption_path).read_text()); raw["affected_ratio"]=a.ratio
for key in ("short_burst_length","long_gap_length","correlated_group_size"): raw[key]=tuple(raw[key])
cc=CorruptionConfig(**raw); root_out=Path(cfg["training"]["output_dir"])/f"ratio_{a.ratio:.2f}"; root_out.mkdir(parents=True,exist_ok=True)
for i in range(a.runs):
    seed=a.start_seed+i; run=root_out/f"seed_{seed}"; run.mkdir(parents=True,exist_ok=True)
    # Independent corruption in train/validation/test, with deterministic seed offsets.
    chunks=[]; bounds=chronological_bounds(len(clean["values"]),cfg["data"]["split"])
    for split_idx,(name,(s,e)) in enumerate(bounds.items()):
        part=apply_corruption(clean["values"][s:e],clean["reference_times"][s:e],clean["natural_mask"][s:e],cc,seed+split_idx,graph)
        part.pop("metadata_json",None); chunks.append(part)
    corrupted={k:np.concatenate([c[k] for c in chunks],axis=0) for k in chunks[0]}
    corrupted["metadata_json"]=np.asarray(yaml.safe_dump({"base_seed":seed,"ratio":a.ratio,"split_seed_offsets":[0,1,2]}))
    data_path=run/"corrupted.npz"; save_npz(data_path,**corrupted)
    resolved=load_config(a.config); resolved["project"]["seed"]=seed; resolved["data"]["path"]=str(data_path); resolved["training"]["output_dir"]=str(run)
    conf=run/"config.yaml"; save_config(resolved,conf)
    commands=[["python","scripts/train.py","--config",str(conf)],
              ["python","scripts/evaluate.py","--config",str(conf),"--checkpoint",str(run/"best.pt"),"--output",str(run/"metrics.json")]]
    if a.device:
        commands[0]+=["--device",a.device]; commands[1]+=["--device",a.device]
    print(f"run={i+1}/{a.runs} seed={seed} data={data_path}")
    if not a.dry_run:
        for cmd in commands: subprocess.run(cmd,check=True,cwd=ROOT)
print(root_out)
