from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn


class ReconstructionHead(nn.Module):
    """Attach a shared reconstruction head to a representation-producing backbone."""
    def __init__(self, backbone: nn.Module, representation_dim: int, channels: int):
        super().__init__(); self.backbone=backbone; self.head=nn.Linear(representation_dim,channels)
    def forward(self,*args: Any,**kwargs: Any) -> torch.Tensor:
        rep=self.backbone(*args,**kwargs)
        if isinstance(rep,dict): rep=rep.get("representation",rep.get("hidden",rep.get("prediction")))
        if isinstance(rep,(tuple,list)): rep=rep[0]
        return self.head(rep)


class PredictionBackboneAdapter(nn.Module):
    """Normalize heterogeneous prediction backbones to a reconstruction `prediction` output."""
    def __init__(self, backbone: nn.Module): super().__init__(); self.backbone=backbone
    def forward(self,*args: Any,**kwargs: Any) -> dict[str,torch.Tensor]:
        out=self.backbone(*args,**kwargs)
        if isinstance(out,dict):
            for key in ("prediction","reconstruction","forecast","output"):
                if key in out: return {"prediction":out[key]}
            raise KeyError("Backbone dictionary has no recognized estimation output")
        if isinstance(out,(tuple,list)): out=out[0]
        return {"prediction":out}


@dataclass(frozen=True)
class AdaptationRecord:
    name: str
    upstream_url: str
    upstream_commit: str
    removed_stages: tuple[str,...]
    added_stages: tuple[str,...]
    notes: str=""

    def validate(self) -> None:
        if self.upstream_commit in {"", "AUTHOR_REQUIRED"}:
            raise ValueError(f"{self.name}: upstream commit is not pinned")
