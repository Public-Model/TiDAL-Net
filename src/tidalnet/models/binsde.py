from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)


class BiNSDECompensator(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 32, embedding_dim: int = 8,
                 steps: int = 4, mc_paths: int = 4, diffusion_floor: float = 1e-4,
                 dropout: float = 0.0, use_diffusion: bool = True,
                 uncertainty_fusion: bool = True, neighbor_conditioning: bool = True,
                 forward_only: bool = False):
        super().__init__()
        self.channels=channels; self.hidden_dim=hidden_dim; self.steps=max(1,steps)
        self.mc_paths=max(1,mc_paths); self.diffusion_floor=diffusion_floor
        self.use_diffusion=use_diffusion; self.uncertainty_fusion=uncertainty_fusion
        self.neighbor_conditioning=neighbor_conditioning; self.forward_only=forward_only
        self.channel_embedding = nn.Embedding(channels, embedding_dim)
        self.value_encoder = nn.Linear(1, hidden_dim)
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        inp = hidden_dim*2 + embedding_dim + 1
        self.drift_f = MLP(inp, hidden_dim, hidden_dim, dropout)
        self.drift_b = MLP(inp, hidden_dim, hidden_dim, dropout)
        self.diff_f = MLP(hidden_dim+embedding_dim+1, hidden_dim, hidden_dim, dropout)
        self.diff_b = MLP(hidden_dim+embedding_dim+1, hidden_dim, hidden_dim, dropout)
        self.decoder = nn.Linear(hidden_dim, 1)
        self.gate = MLP(3, 1, max(8, hidden_dim//2), dropout)

    def _neighbor(self, state: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        # state [K,B,M,H], graph [M,M]
        if not self.neighbor_conditioning:
            return torch.zeros_like(state)
        norm = graph / graph.sum(-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("ij,kbjh->kbih", norm, state)

    def _propagate(self, x: torch.Tensor, mask: torch.Tensor, obs_times: torch.Tensor,
                   graph: torch.Tensor, reverse: bool):
        if reverse:
            x=x.flip(1); mask=mask.flip(1); obs_times=obs_times.flip(1)
        b,l,m=x.shape; k=self.mc_paths; h=self.hidden_dim
        ids=torch.arange(m, device=x.device); emb=self.channel_embedding(ids)
        emb_h=self.embedding_proj(emb).view(1,m,h)
        anchor=self.value_encoder(x.unsqueeze(-1)) + emb_h.unsqueeze(1)
        state=anchor[:,0].unsqueeze(0).expand(k,-1,-1,-1).contiguous()
        means=[]; vars_=[]; drifts=[]
        drift_net=self.drift_b if reverse else self.drift_f
        diff_net=self.diff_b if reverse else self.diff_f
        prev_time=obs_times[:,0]
        for t in range(l):
            current_time=obs_times[:,t]
            dt=(current_time-prev_time).abs().clamp_min(1e-4)
            ds=dt/self.steps
            drift_acc=0.0
            for _ in range(self.steps):
                context=self._neighbor(state, graph)
                emb_k=emb.view(1,1,m,-1).expand(k,b,-1,-1)
                ds_k=ds.view(1,b,m,1).expand(k,-1,-1,-1)
                drift=drift_net(torch.cat([state,context,emb_k,ds_k],dim=-1))
                if self.use_diffusion:
                    diff=F.softplus(diff_net(torch.cat([state,emb_k,ds_k],dim=-1)))+self.diffusion_floor
                    noise=torch.randn_like(state)
                    state=state+drift*ds_k+diff*torch.sqrt(ds_k)*noise
                else:
                    state=state+drift*ds_k
                drift_acc=drift_acc+drift
            observed=mask[:,t].view(1,b,m,1)
            state=observed*anchor[:,t].unsqueeze(0)+(1-observed)*state
            means.append(state.mean(0)); vars_.append(state.var(0,unbiased=False));
            drifts.append((drift_acc/self.steps).mean(0))
            prev_time=current_time
        mean=torch.stack(means,1); var=torch.stack(vars_,1); drift=torch.stack(drifts,1)
        if reverse: mean=mean.flip(1); var=var.flip(1); drift=drift.flip(1)
        return mean,var,drift

    @staticmethod
    def _distance(mask: torch.Tensor, times: torch.Tensor, reverse: bool=False) -> torch.Tensor:
        if reverse: mask=mask.flip(1); times=times.flip(1)
        b,l,m=mask.shape; dist=[]; last=times[:,0]
        for t in range(l):
            d=(times[:,t]-last).abs()
            last=torch.where(mask[:,t]>0, times[:,t], last)
            dist.append(d)
        out=torch.stack(dist,1)
        return out.flip(1) if reverse else out

    def forward(self, x: torch.Tensor, mask: torch.Tensor, reference_times: torch.Tensor,
                offsets: torch.Tensor, graph: torch.Tensor) -> dict[str, torch.Tensor]:
        if reference_times.ndim==2:
            reference_times=reference_times.unsqueeze(-1).expand_as(x)
        obs_times=reference_times+offsets
        fm,fv,fd=self._propagate(x,mask,obs_times,graph,False)
        if self.forward_only:
            bm,bv,bd=fm,fv,fd
        else:
            bm,bv,bd=self._propagate(x,mask,obs_times,graph,True)
        fval=self.decoder(fm).squeeze(-1); bval=self.decoder(bm).squeeze(-1)
        fu=fv.mean(-1); bu=bv.mean(-1)
        if self.uncertainty_fusion:
            df=self._distance(mask,obs_times,False); db=self._distance(mask,obs_times,True)
            wf=1.0/(fu+df+1e-5); wb=1.0/(bu+db+1e-5)
            estimate=(wf*fval+wb*bval)/(wf+wb).clamp_min(1e-6)
            uncertainty=1.0/(wf+wb).clamp_min(1e-6)
            latent=(wf.unsqueeze(-1)*fm+wb.unsqueeze(-1)*bm)/(wf+wb).unsqueeze(-1).clamp_min(1e-6)
        else:
            estimate=0.5*(fval+bval); uncertainty=0.5*(fu+bu); latent=0.5*(fm+bm)
        severity=torch.stack([1-mask, offsets.abs(), uncertainty],dim=-1)
        residual_gate=torch.sigmoid(self.gate(severity).squeeze(-1))
        residual_gate=torch.where(mask<=0, torch.ones_like(residual_gate), residual_gate)
        compensated=x+residual_gate*(estimate-x)
        return {"compensated":compensated,"estimate":estimate,"uncertainty":uncertainty,
                "latent":latent,"drift":0.5*(fd+bd),"residual_gate":residual_gate}
