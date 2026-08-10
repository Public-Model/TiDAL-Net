from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class LiquidGRUCell(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, tau_min: float=0.05,
                 dynamic_tau: bool=True, dropout: float=0.0):
        super().__init__(); self.hidden_dim=hidden_dim; self.tau_min=tau_min; self.dynamic_tau=dynamic_tau
        # value, dt, delta, uncertainty, channel embedding, previous hidden, graph feedback
        in_dim=4+embedding_dim+2*hidden_dim
        self.feature=nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.SiLU(),nn.Dropout(dropout))
        self.z=nn.Linear(2*hidden_dim,hidden_dim); self.r=nn.Linear(2*hidden_dim,hidden_dim)
        self.candidate=nn.Linear(2*hidden_dim,hidden_dim)
        self.tau_net=nn.Linear(2*hidden_dim,3*hidden_dim)
        self.fixed_tau=nn.Parameter(torch.zeros(3,hidden_dim))
        self.indicator=nn.Sequential(nn.Linear(hidden_dim+2,hidden_dim//2 or 1),nn.SiLU(),nn.Linear(hidden_dim//2 or 1,1))

    def forward(self, value: torch.Tensor, dt: torch.Tensor, delta: torch.Tensor,
                uncertainty: torch.Tensor, embedding: torch.Tensor, hidden: torch.Tensor,
                graph_feedback: torch.Tensor):
        raw=torch.cat([value.unsqueeze(-1),dt.unsqueeze(-1),delta.unsqueeze(-1),
                       uncertainty.unsqueeze(-1),embedding,hidden,graph_feedback],-1)
        feat=self.feature(raw); joint=torch.cat([feat,hidden],-1)
        if self.dynamic_tau: tau=F.softplus(self.tau_net(joint)).view(*joint.shape[:-1],3,self.hidden_dim)+self.tau_min
        else: tau=F.softplus(self.fixed_tau).view(1,1,3,self.hidden_dim)+self.tau_min
        tz,tr,th=tau.unbind(-2)
        z=torch.sigmoid(self.z(joint)/tz); r=torch.sigmoid(self.r(joint)/tr)
        cand=torch.tanh(self.candidate(torch.cat([feat,r*hidden],-1)))
        rate=1-torch.exp(-dt.unsqueeze(-1).abs().clamp_min(1e-4)/th)
        new=hidden+rate*z*(cand-hidden)
        ind=F.softplus(self.indicator(torch.cat([new,uncertainty.unsqueeze(-1),delta.abs().unsqueeze(-1)],-1)))
        return new,ind,tau
