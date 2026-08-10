#!/usr/bin/env python
from pathlib import Path
import argparse,json,pandas as pd
from tidalnet.evaluation.statistics import paired_summary
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); a=p.parse_args()
rows=[]
for f in Path(a.input).rglob("metrics.json"):
    r=json.loads(f.read_text()); r["source"]=str(f); rows.append(r)
if not rows: print("No metrics.json files found; nothing aggregated."); raise SystemExit(0)
df=pd.DataFrame(rows); summary=[]
for (model,dataset,ratio),g in df.groupby(["model","dataset","ratio"]):
    for metric in ["mae","rmse","mape"]:
        s=paired_summary(g[metric].values); summary.append({"model":model,"dataset":dataset,"ratio":ratio,"metric":metric,**s})
Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(summary).to_csv(a.output,index=False); print(a.output)
