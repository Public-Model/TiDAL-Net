import torch
from tidalnet.models import TiDALNet

def test_model_shapes_and_backward():
    torch.manual_seed(1); b,l,m=2,6,4
    model=TiDALNet(m,hidden_dim=8,channel_embedding_dim=4,nsde_steps=1,mc_paths=2,graph_heads=1,max_graph_radius=2,base_temporal_window=2,dropout=0.0)
    x=torch.randn(b,l,m); mask=torch.ones_like(x); mask[:,2:4,1]=0
    times=torch.arange(l).float().view(1,l).expand(b,-1); offsets=torch.zeros_like(x); graph=torch.eye(m); graph[0,1]=graph[1,0]=.8
    out=model(x,mask,times,offsets,graph)
    assert out["prediction"].shape==(b,l,m)
    assert out["uncertainty"].shape==(b,l,m)
    out["prediction"].mean().backward()
    assert any(p.grad is not None for p in model.parameters())
