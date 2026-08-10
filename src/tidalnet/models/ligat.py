from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def _hop_distance(graph: torch.Tensor) -> torch.Tensor:
    m=graph.shape[0]; inf=m+1
    d=torch.full((m,m),inf,dtype=torch.long,device=graph.device)
    d[graph>0]=1; d.fill_diagonal_(0)
    for k in range(m): d=torch.minimum(d,d[:,k:k+1]+d[k:k+1,:])
    return d


class LiquidGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int=2, max_radius: int=3,
                 base_window: int=4, dynamic_radius: bool=True, dynamic_window: bool=True,
                 dropout: float=0.0):
        super().__init__(); self.hidden_dim=hidden_dim; self.heads=heads
        self.max_radius=max_radius; self.base_window=max(1,base_window)
        self.dynamic_radius=dynamic_radius; self.dynamic_window=dynamic_window
        self.q=nn.Linear(hidden_dim,hidden_dim*heads,bias=False)
        self.k=nn.Linear(hidden_dim,hidden_dim*heads,bias=False)
        self.v=nn.Linear(hidden_dim,hidden_dim*heads,bias=False)
        self.out=nn.Linear(hidden_dim*heads,hidden_dim)
        self.radius_net=nn.Linear(2,1); self.window_net=nn.Linear(2,1)
        self.dropout=nn.Dropout(dropout)

    def _temporal_summary(self, history: torch.Tensor, indicator: torch.Tensor,
                          uncertainty: torch.Tensor) -> torch.Tensor:
        # history [B,T,M,H]; node-specific hard window, deterministic and auditable
        b,t,m,h=history.shape
        if self.dynamic_window:
            scale=torch.sigmoid(self.window_net(torch.cat([indicator,uncertainty.unsqueeze(-1)],-1)))
            windows=1+(scale.squeeze(-1)*max(self.base_window-1,0)).round().long()
        else: windows=torch.full((b,m),self.base_window,device=history.device,dtype=torch.long)
        ages=torch.arange(t-1,-1,-1,device=history.device).view(1,t,1)
        valid=ages < windows.unsqueeze(1)
        return (history*valid.unsqueeze(-1)).sum(1)/valid.sum(1).unsqueeze(-1).clamp_min(1)

    def forward(self, hidden: torch.Tensor, history: torch.Tensor, indicator: torch.Tensor,
                uncertainty: torch.Tensor, graph: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        b,m,h=hidden.shape; summary=self._temporal_summary(history,indicator,uncertainty)
        hop=_hop_distance(graph)
        if self.dynamic_radius:
            scale=torch.sigmoid(self.radius_net(torch.cat([indicator,uncertainty.unsqueeze(-1)],-1)))
            radius=1+(scale.squeeze(-1)*max(self.max_radius-1,0)).round().long()
        else: radius=torch.full((b,m),self.max_radius,device=hidden.device,dtype=torch.long)
        support=(hop.view(1,m,m)<=radius.unsqueeze(-1)) & (graph.view(1,m,m)>0)
        q=self.q(hidden).view(b,m,self.heads,h).permute(0,2,1,3)
        k=self.k(summary).view(b,m,self.heads,h).permute(0,2,1,3)
        v=self.v(summary).view(b,m,self.heads,h).permute(0,2,1,3)
        logits=torch.einsum("bhid,bhjd->bhij",q,k)/(h**0.5)
        prior=torch.log(graph.clamp_min(1e-8)).view(1,1,m,m)
        temporal=-(indicator.unsqueeze(2)-indicator.unsqueeze(1)).abs().squeeze(-1).unsqueeze(1)
        cosine=F.cosine_similarity(hidden.unsqueeze(2),summary.unsqueeze(1),dim=-1).unsqueeze(1)
        logits=logits+prior+temporal+cosine
        logits=logits.masked_fill(~support.unsqueeze(1),torch.finfo(logits.dtype).min)
        attn=self.dropout(torch.softmax(logits,-1))
        fused=torch.einsum("bhij,bhjd->bhid",attn,v).permute(0,2,1,3).reshape(b,m,self.heads*h)
        fused=self.out(fused)
        return hidden+fused,attn
