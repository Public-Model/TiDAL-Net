from __future__ import annotations
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from .losses import reconstruction_loss
from tidalnet.evaluation.metrics import metrics_torch


def _move(batch,device): return {k:v.to(device) for k,v in batch.items()}


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, graph: torch.Tensor,
                config: dict, device: torch.device) -> dict:
    model.to(device); graph=graph.to(device)
    tc=config["training"]
    optimizer=torch.optim.Adam(model.parameters(),lr=tc["learning_rate"],weight_decay=tc.get("weight_decay",0.0))
    weights={"mae":tc.get("masked_mae_weight",1.0),"mse":tc.get("masked_mse_weight",1.0),
             "identity":tc.get("identity_weight",0.0),"cross_channel":tc.get("cross_channel_weight",0.0)}
    outdir=Path(tc["output_dir"]); outdir.mkdir(parents=True,exist_ok=True)
    best=float("inf"); patience=0; history=[]
    for epoch in range(int(tc["epochs"])):
        model.train(); train_losses=[]
        for batch in train_loader:
            batch=_move(batch,device); optimizer.zero_grad(set_to_none=True)
            out=model(batch["received"],batch["mask"],batch["times"],batch["offsets"],graph)
            loss,parts=reconstruction_loss(out,batch["target"],batch["eval_mask"],batch["mask"],graph,weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),tc.get("grad_clip",1.0)); optimizer.step()
            train_losses.append(parts["loss"])
        val=evaluate_model(model,val_loader,graph,device)
        record={"epoch":epoch+1,"train_loss":sum(train_losses)/max(len(train_losses),1),**{f"val_{k}":v for k,v in val.items()}}
        history.append(record)
        if val["mae"]<best:
            best=val["mae"]; patience=0
            torch.save({"model":model.state_dict(),"config":config,"epoch":epoch+1,"val":val},outdir/"best.pt")
        else:
            patience+=1
            if patience>=int(tc["early_stopping_patience"]): break
    (outdir/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    return {"best_val_mae":best,"epochs_ran":len(history)}


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, graph: torch.Tensor, device: torch.device) -> dict[str,float]:
    model.eval(); graph=graph.to(device); sums={"mae":0.,"rmse":0.,"mape":0.}; n=0
    for batch in loader:
        batch=_move(batch,device); out=model(batch["received"],batch["mask"],batch["times"],batch["offsets"],graph)
        met=metrics_torch(out["prediction"],batch["target"],batch["eval_mask"])
        for k in sums: sums[k]+=met[k]
        n+=1
    return {k:v/max(n,1) for k,v in sums.items()}
