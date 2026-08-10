from __future__ import annotations
import numpy as np
import torch


def metrics_numpy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, eps: float=1e-6) -> dict[str,float]:
    sel=mask>0
    if not np.any(sel): return {"mae":float("nan"),"rmse":float("nan"),"mape":float("nan")}
    e=pred[sel]-target[sel]
    return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),
            "mape":float(np.mean(np.abs(e)/(np.abs(target[sel])+eps)))}


def metrics_torch(pred: torch.Tensor,target: torch.Tensor,mask: torch.Tensor,eps: float=1e-6) -> dict[str,float]:
    denom=mask.sum().clamp_min(1.0); e=pred-target
    mae=(e.abs()*mask).sum()/denom; rmse=torch.sqrt((e.pow(2)*mask).sum()/denom)
    mape=((e.abs()/(target.abs()+eps))*mask).sum()/denom
    return {"mae":float(mae),"rmse":float(rmse),"mape":float(mape)}
