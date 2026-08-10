#!/usr/bin/env python
import argparse,torch,numpy as np
from torch.utils.data import DataLoader
from _common import load_experiment
from tidalnet.utils import device_from_config
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True); a=p.parse_args()
cfg,data,graph,sets,model=load_experiment(a.config); device=device_from_config(); ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck["model"]); model.to(device).eval(); graph=graph.to(device)
batch=next(iter(DataLoader(sets["test"],batch_size=1))); batch={k:v.to(device) for k,v in batch.items()}
with torch.no_grad(): out=model(batch["received"],batch["mask"],batch["times"],batch["offsets"],graph)
np.savez_compressed(a.output,prediction=out["prediction"].cpu().numpy(),target=batch["target"].cpu().numpy(),eval_mask=batch["eval_mask"].cpu().numpy()); print(a.output)
