from __future__ import annotations
import numpy as np
from scipy import stats


def paired_summary(a, b=None, confidence: float=0.95) -> dict[str,float]:
    a=np.asarray(a,dtype=float); a=a[np.isfinite(a)]
    n=len(a); mean=float(np.mean(a)); std=float(np.std(a,ddof=1)) if n>1 else 0.0
    sem=std/np.sqrt(max(n,1)); q=stats.t.ppf((1+confidence)/2,max(n-1,1)) if n>1 else 0.0
    out={"n":n,"mean":mean,"std":std,"ci_low":mean-q*sem,"ci_high":mean+q*sem}
    if b is not None:
        b=np.asarray(b,dtype=float); valid=np.isfinite(a)&np.isfinite(b[:len(a)])
        stat,p=stats.ttest_rel(a[valid],b[:len(a)][valid]) if valid.sum()>1 else (np.nan,np.nan)
        out.update({"paired_t":float(stat),"paired_p":float(p)})
    return out
