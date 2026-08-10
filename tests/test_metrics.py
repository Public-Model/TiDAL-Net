import numpy as np
from tidalnet.evaluation.metrics import metrics_numpy

def test_metrics_only_use_mask():
    p=np.array([0.,100.]); y=np.array([1.,0.]); m=np.array([1.,0.])
    r=metrics_numpy(p,y,m); assert r["mae"]==1.0 and r["rmse"]==1.0
