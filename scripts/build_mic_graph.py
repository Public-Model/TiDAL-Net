#!/usr/bin/env python
from pathlib import Path
import argparse,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tidalnet.data.io import load_npz
from tidalnet.graph.mic import approximate_mic_matrix,build_candidate_graph
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--top-k",type=int,default=5); p.add_argument("--threshold",type=float,default=0.1); p.add_argument("--max-bins",type=int,default=8); p.add_argument("--train-fraction",type=float,default=0.6); a=p.parse_args()
d=load_npz(a.input); n=int(len(d["values"])*a.train_fraction)
mic=approximate_mic_matrix(d["values"][:n],d["natural_mask"][:n],a.max_bins); graph=build_candidate_graph(mic,a.top_k,a.threshold)
Path(a.output).parent.mkdir(parents=True,exist_ok=True); np.save(a.output,graph); np.save(str(Path(a.output).with_suffix(''))+'_mic.npy',mic); print(a.output)
