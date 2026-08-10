#!/usr/bin/env python
import argparse,torch,json
from pathlib import Path
from torch.utils.data import DataLoader
from _common import load_experiment
from tidalnet.training.engine import evaluate_model
from tidalnet.utils import device_from_config
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--device"); p.add_argument("--output"); a=p.parse_args()
cfg,data,graph,sets,model=load_experiment(a.config); device=device_from_config(a.device); ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck["model"]); model.to(device)
loader=DataLoader(sets["test"],batch_size=cfg["training"]["batch_size"]); metrics=evaluate_model(model,loader,graph,device); print(metrics)
path=Path(a.output) if a.output else Path(a.checkpoint).parent/"metrics.json"; path.write_text(json.dumps(metrics,indent=2)+"\n")
