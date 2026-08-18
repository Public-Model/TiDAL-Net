#!/usr/bin/env python
"""Generate one auditable corruption file from a complete telemetry NPZ."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tidalnet.data.io import load_npz, save_npz
from tidalnet.data.corruption import apply_corruption, resolve_corruption_config
from tidalnet.data.windows import chronological_bounds


p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--graph")
p.add_argument("--config", default="configs/corruption/reality_informed.yaml")
p.add_argument("--dataset", help="Optional dataset profile name, e.g. esa_adb, real_a, real_b")
p.add_argument("--ratio", type=float)
p.add_argument("--seed", type=int, default=2026)
p.add_argument(
    "--global-mask",
    action="store_true",
    help="Generate one mask over the whole sequence instead of independently within chronological splits",
)
a = p.parse_args()

raw = yaml.safe_load(Path(a.config).read_text())
cfg = resolve_corruption_config(raw, affected_ratio=a.ratio, dataset=a.dataset)
d = load_npz(a.input)
graph = np.load(a.graph) if a.graph else None

if a.global_mask:
    out = apply_corruption(
        d["values"], d["reference_times"], d["natural_mask"], cfg, a.seed, graph
    )
else:
    bounds = chronological_bounds(len(d["values"]), (0.6, 0.2, 0.2))
    chunks = []
    split_meta = {}
    for idx, (name, (start, end)) in enumerate(bounds.items()):
        chunk = apply_corruption(
            d["values"][start:end],
            d["reference_times"][start:end],
            d["natural_mask"][start:end],
            cfg,
            a.seed + idx,
            graph,
        )
        split_meta[name] = json.loads(str(chunk.pop("metadata_json")))
        chunks.append(chunk)
    out = {key: np.concatenate([c[key] for c in chunks], axis=0) for key in chunks[0]}
    out["metadata_json"] = np.asarray(
        json.dumps(
            {
                "mode": "split_wise",
                "dataset_profile": a.dataset,
                "base_seed": a.seed,
                "resolved_config": cfg.__dict__,
                "splits": split_meta,
            }
        )
    )

save_npz(a.output, **out)
print(a.output)
print(str(out["metadata_json"]))
