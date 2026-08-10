#!/usr/bin/env python
from pathlib import Path
import argparse,subprocess,json
p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=["nasa","esa-adb"],required=True); p.add_argument("--root",default="data/raw"); p.add_argument("--force",action="store_true"); a=p.parse_args()
root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
if a.dataset=="nasa": url="https://github.com/khundman/telemanom.git"; dst=root/"telemanom"
else: url="https://github.com/kplabs-pl/ESA-ADB.git"; dst=root/"ESA-ADB"
if dst.exists() and not a.force: print(f"Exists: {dst}"); raise SystemExit(0)
if dst.exists(): import shutil; shutil.rmtree(dst)
subprocess.run(["git","clone","--depth","1",url,str(dst)],check=True)
commit=subprocess.check_output(["git","-C",str(dst),"rev-parse","HEAD"],text=True).strip()
(dst/"TIDALNET_UPSTREAM.json").write_text(json.dumps({"url":url,"commit":commit},indent=2)+"\n")
print(dst)
