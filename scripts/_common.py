from __future__ import annotations
from pathlib import Path
import sys, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path: sys.path.insert(0,str(ROOT/"src"))
from tidalnet.config import load_config
from tidalnet.data.io import load_npz
from tidalnet.data.normalization import ZScoreNormalizer
from tidalnet.data.windows import TelemetryWindowDataset, chronological_bounds
from tidalnet.models import TiDALNet


def load_experiment(config_path):
    cfg=load_config(config_path); data=load_npz(cfg["data"]["path"])
    bounds=chronological_bounds(len(data["values"]),cfg["data"]["split"])
    # Leakage-safe normalization: fit only on naturally valid training observations.
    if cfg["data"].get("normalize","none")=="zscore":
        a,b=bounds["train"]
        normalizer=ZScoreNormalizer().fit(data["values"][a:b],data["natural_mask"][a:b])
        data["values"]=normalizer.transform(data["values"])
        received=normalizer.transform(data.get("received_values",data["values"]))
        # Artificially missing entries must remain the neutral zero after normalization.
        obs=data.get("observation_mask",data["natural_mask"])
        data["received_values"]=(received*obs).astype(np.float32)
        data["normalization_mean"]=normalizer.mean.astype(np.float32)
        data["normalization_std"]=normalizer.std.astype(np.float32)
    graph=torch.as_tensor(np.load(cfg["data"]["graph_path"]),dtype=torch.float32)
    sets={k:TelemetryWindowDataset(data,*v,cfg["data"]["window"],cfg["data"]["stride"]) for k,v in bounds.items()}
    channels=data["values"].shape[1]; mc=cfg["model"]
    model=TiDALNet(channels=channels,hidden_dim=mc["hidden_dim"],channel_embedding_dim=mc["channel_embedding_dim"],
       nsde_steps=mc["nsde_steps"],mc_paths=mc.get("mc_paths_train",4),diffusion_floor=mc["diffusion_floor"],
       tau_min=mc["tau_min"],max_graph_radius=mc["max_graph_radius"],base_temporal_window=mc["base_temporal_window"],
       graph_heads=mc["graph_heads"],dropout=mc["dropout"],use_diffusion=mc.get("use_diffusion",True),
       use_uncertainty_fusion=mc.get("use_uncertainty_fusion",True),use_neighbor_conditioning=mc.get("use_neighbor_conditioning",True),
       use_dynamic_tau=mc.get("use_dynamic_tau",True),use_dynamic_radius=mc.get("use_dynamic_radius",True),
       use_dynamic_window=mc.get("use_dynamic_window",True),use_graph_feedback=mc.get("use_graph_feedback",True),
       forward_only=mc.get("forward_only",False),disable_binsde=mc.get("disable_binsde",False),
       disable_ligru=mc.get("disable_ligru",False),disable_ligat=mc.get("disable_ligat",False))
    return cfg,data,graph,sets,model
