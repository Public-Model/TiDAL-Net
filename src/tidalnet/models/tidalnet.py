from __future__ import annotations
import torch
from torch import nn
from .binsde import BiNSDECompensator
from .ligru import LiquidGRUCell
from .ligat import LiquidGraphAttention


class TiDALNet(nn.Module):
    def __init__(self, channels: int, hidden_dim: int=32, channel_embedding_dim: int=8,
                 nsde_steps: int=4, mc_paths: int=4, diffusion_floor: float=1e-4,
                 tau_min: float=0.05, max_graph_radius: int=3, base_temporal_window: int=4,
                 graph_heads: int=2, dropout: float=0.1, use_diffusion: bool=True,
                 use_uncertainty_fusion: bool=True, use_neighbor_conditioning: bool=True,
                 use_dynamic_tau: bool=True, use_dynamic_radius: bool=True,
                 use_dynamic_window: bool=True, use_graph_feedback: bool=True,
                 forward_only: bool=False, disable_binsde: bool=False,
                 disable_ligru: bool=False, disable_ligat: bool=False):
        super().__init__(); self.channels=channels; self.hidden_dim=hidden_dim
        self.use_graph_feedback=use_graph_feedback; self.disable_binsde=disable_binsde
        self.disable_ligru=disable_ligru; self.disable_ligat=disable_ligat
        self.channel_embedding=nn.Embedding(channels,channel_embedding_dim)
        self.binsde=BiNSDECompensator(channels,hidden_dim,channel_embedding_dim,nsde_steps,mc_paths,
                                     diffusion_floor,dropout,use_diffusion,use_uncertainty_fusion,
                                     use_neighbor_conditioning,forward_only)
        self.ligru=LiquidGRUCell(hidden_dim,channel_embedding_dim,tau_min,use_dynamic_tau,dropout)
        self.input_proj=nn.Linear(1,hidden_dim)
        self.ligat=LiquidGraphAttention(hidden_dim,graph_heads,max_graph_radius,base_temporal_window,
                                        use_dynamic_radius,use_dynamic_window,dropout)
        self.decoder=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))

    def forward(self, received: torch.Tensor, mask: torch.Tensor, times: torch.Tensor,
                offsets: torch.Tensor, graph: torch.Tensor) -> dict[str,torch.Tensor]:
        b,l,m=received.shape
        if self.disable_binsde:
            compensated=received; uncertainty=torch.zeros_like(received)
            drift=torch.zeros(b,l,m,self.hidden_dim,device=received.device)
            residual_gate=torch.zeros_like(received)
        else:
            comp=self.binsde(received,mask,times,offsets,graph)
            compensated=comp["compensated"]; uncertainty=comp["uncertainty"]
            drift=comp["drift"]; residual_gate=comp["residual_gate"]
        ids=torch.arange(m,device=received.device); emb=self.channel_embedding(ids).view(1,m,-1).expand(b,-1,-1)
        hidden=torch.zeros(b,m,self.hidden_dim,device=received.device)
        feedback=torch.zeros_like(hidden); history=[]; outputs=[]; indicators=[]; taus=[]; attentions=[]
        prev_value=compensated[:,0]
        for t in range(l):
            dt=torch.ones(b,m,device=received.device) if t==0 else (times[:,t]-times[:,t-1]).abs().unsqueeze(-1).expand(-1,m)
            delta=compensated[:,t]-prev_value
            if self.disable_ligru:
                hidden=self.input_proj(compensated[:,t].unsqueeze(-1))
                indicator=delta.abs().unsqueeze(-1); tau=torch.ones(b,m,3,self.hidden_dim,device=received.device)
            else:
                hidden,indicator,tau=self.ligru(compensated[:,t],dt,delta,uncertainty[:,t],emb,hidden,feedback)
            history.append(hidden); hist=torch.stack(history,1)
            if not self.disable_ligat:
                hidden,attn=self.ligat(hidden,hist,indicator,uncertainty[:,t],graph)
                feedback=hidden if self.use_graph_feedback else torch.zeros_like(hidden)
                attentions.append(attn)
            outputs.append(compensated[:,t]+self.decoder(hidden).squeeze(-1))
            indicators.append(indicator); taus.append(tau); prev_value=compensated[:,t]
        prediction=torch.stack(outputs,1)
        return {"prediction":prediction,"compensated":compensated,"uncertainty":uncertainty,
                "drift":drift,"residual_gate":residual_gate,"indicator":torch.stack(indicators,1),
                "tau":torch.stack(taus,1),"attention":attentions}
