import numpy as np
from tidalnet.data.normalization import ZScoreNormalizer

def test_normalizer_uses_requested_training_slice_only():
    x=np.array([[0.,0.],[2.,2.],[100.,100.]],dtype=float); m=np.ones_like(x)
    n=ZScoreNormalizer().fit(x[:2],m[:2])
    assert np.allclose(n.mean,[1.,1.])
    assert np.allclose(n.std,[1.,1.])
