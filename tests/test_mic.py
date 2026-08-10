import numpy as np
from tidalnet.graph.mic import approximate_mic_matrix,build_candidate_graph

def test_mic_graph_is_symmetric_and_training_only_compatible():
    rng=np.random.default_rng(1); x=rng.normal(size=(100,3)); x[:,1]=x[:,0]+rng.normal(0,.01,100)
    mic=approximate_mic_matrix(x,np.ones_like(x),max_bins=4); g=build_candidate_graph(mic,top_k=1,threshold=.0)
    assert mic[0,1] > mic[0,2]
    assert np.allclose(g,g.T)
    assert np.all(np.diag(g)==1)
