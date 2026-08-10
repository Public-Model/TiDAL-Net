from __future__ import annotations
import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x*mask).sum()/mask.sum().clamp_min(1.0)


def cross_channel_consistency(drift: torch.Tensor, graph: torch.Tensor,
                              reliability: torch.Tensor) -> torch.Tensor:
    # Projected/lag-aware exact paper loss requires final author parameters; this shared-space
    # implementation is the runnable review default and preserves reliability and graph weighting.
    diff=(drift.unsqueeze(3)-drift.unsqueeze(2)).pow(2).mean(-1)
    weight=graph.view(1,1,*graph.shape)*reliability.unsqueeze(3)*reliability.unsqueeze(2)
    return (diff*weight).sum()/weight.sum().clamp_min(1.0)


def reconstruction_loss(output: dict[str,torch.Tensor], target: torch.Tensor,
                        eval_mask: torch.Tensor, observation_mask: torch.Tensor,
                        graph: torch.Tensor, weights: dict[str,float]) -> tuple[torch.Tensor,dict[str,float]]:
    pred=output["prediction"]; err=pred-target
    mae=masked_mean(err.abs(),eval_mask); mse=masked_mean(err.pow(2),eval_mask)
    untouched=(observation_mask*(1-eval_mask)).clamp(0,1)
    identity=masked_mean(err.abs(),untouched)
    cc=cross_channel_consistency(output["drift"],graph,observation_mask)
    total=weights.get("mae",1.0)*mae+weights.get("mse",1.0)*mse+weights.get("identity",0.0)*identity+weights.get("cross_channel",0.0)*cc
    return total,{"loss":float(total.detach()),"mae":float(mae.detach()),"mse":float(mse.detach()),
                  "identity":float(identity.detach()),"cross_channel":float(cc.detach())}
