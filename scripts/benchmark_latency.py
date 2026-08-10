#!/usr/bin/env python
import argparse,time,torch,numpy as np
from torch.utils.data import DataLoader
from _common import load_experiment
from tidalnet.utils import device_from_config
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint"); p.add_argument("--device"); a=p.parse_args()
cfg,data,graph,sets,model=load_experiment(a.config); device=device_from_config(a.device); model.to(device).eval(); graph=graph.to(device)
if a.checkpoint: model.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=False)["model"])
batch=next(iter(DataLoader(sets["test"],batch_size=1))); batch={k:v.to(device) for k,v in batch.items()}; lc=cfg["latency"]
def sync():
    if device.type=="cuda" and lc.get("synchronize_cuda",True): torch.cuda.synchronize()
with torch.no_grad():
    for _ in range(int(lc["warmup"])): model(batch["received"],batch["mask"],batch["times"],batch["offsets"],graph)
    times=[]
    for _ in range(int(lc["repetitions"])):
        sync(); start=time.perf_counter(); model(batch["received"],batch["mask"],batch["times"],batch["offsets"],graph); sync(); times.append((time.perf_counter()-start)*1000)
print({"unit":"single_window_batch_1","mean_ms":float(np.mean(times)),"std_ms":float(np.std(times,ddof=1)) if len(times)>1 else 0.0,"device":str(device),"warmup":lc["warmup"],"repetitions":lc["repetitions"]})
