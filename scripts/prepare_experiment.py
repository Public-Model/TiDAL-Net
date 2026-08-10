#!/usr/bin/env python
"""Prepare immutable corruption manifests for all dataset/ratio/seed combinations."""
from pathlib import Path
import argparse,json,yaml
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--output",default="outputs/manifests/plan.json"); a=p.parse_args()
cfg=yaml.safe_load(Path(a.config).read_text()); start=cfg["seeds"]["start"]; count=cfg["seeds"]["count"]
plan=[]
for ds in cfg["datasets"]:
    for ratio in cfg["ratios"]:
        for seed in range(start,start+count): plan.append({"dataset_config":ds,"ratio":ratio,"seed":seed})
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(plan,indent=2)+"\n"); print(a.output)
