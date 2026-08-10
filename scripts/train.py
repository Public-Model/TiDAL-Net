#!/usr/bin/env python
import argparse,torch,numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from _common import load_experiment
from tidalnet.config import save_config
from tidalnet.training.engine import train_model
from tidalnet.utils import seed_everything,device_from_config
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--device"); a=p.parse_args()
cfg,data,graph,sets,model=load_experiment(a.config); seed_everything(int(cfg["project"]["seed"])); device=device_from_config(a.device)
outdir=Path(cfg["training"]["output_dir"]); outdir.mkdir(parents=True,exist_ok=True); save_config(cfg,outdir/"resolved_config.yaml")
if "normalization_mean" in data: np.savez_compressed(outdir/"normalization.npz",mean=data["normalization_mean"],std=data["normalization_std"])
loaders={k:DataLoader(v,batch_size=cfg["training"]["batch_size"],shuffle=(k=="train")) for k,v in sets.items()}
print(train_model(model,loaders["train"],loaders["val"],graph,cfg,device))
