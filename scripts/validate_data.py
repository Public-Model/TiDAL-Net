#!/usr/bin/env python
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from tidalnet.data.io import load_npz
p=argparse.ArgumentParser(); p.add_argument("path"); a=p.parse_args(); d=load_npz(a.path)
print({k:list(v.shape) for k,v in d.items() if hasattr(v,"shape")})
if "metadata_json" in d: print(json.loads(str(d["metadata_json"])))
