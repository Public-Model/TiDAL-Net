import numpy as np
from tidalnet.data.corruption import CorruptionConfig,apply_corruption
from tidalnet.data.schema import validate_corrupted

def test_corruption_is_deterministic_and_disjoint():
    x=np.arange(400,dtype=np.float32).reshape(100,4); t=np.arange(100,dtype=float); m=np.ones_like(x)
    cfg=CorruptionConfig(affected_ratio=.1,drift_fraction=.4,short_burst_length=(2,4),long_gap_length=(5,10),correlated_group_size=(2,3))
    a=apply_corruption(x,t,m,cfg,123,np.eye(4)); b=apply_corruption(x,t,m,cfg,123,np.eye(4))
    assert np.array_equal(a["artificial_missing_mask"],b["artificial_missing_mask"])
    assert np.array_equal(a["timestamp_offsets"],b["timestamp_offsets"])
    assert not np.any((a["artificial_missing_mask"]>0)&(a["artificial_drift_mask"]>0))
    assert int(a["evaluation_mask"].sum())==40
    validate_corrupted(a)
