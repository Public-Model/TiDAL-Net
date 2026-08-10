from __future__ import annotations
import numpy as np


def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(x), dtype=float)
    return ranks / max(len(x)-1, 1)


def _mi_from_hist(hist: np.ndarray) -> float:
    pxy = hist / max(hist.sum(), 1.0)
    px = pxy.sum(axis=1, keepdims=True); py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    valid = (pxy > 0) & (denom > 0)
    return float(np.sum(pxy[valid] * np.log(pxy[valid] / denom[valid])))


def approximate_mic(x: np.ndarray, y: np.ndarray, max_bins: int = 8) -> float:
    """Deterministic MINE-style finite-grid MIC approximation.

    The exact paper release should pin the same implementation/version used in the experiments.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]; y = y[valid]
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0: return 0.0
    x = _rank01(x); y = _rank01(y)
    best = 0.0
    for bx in range(2, max_bins+1):
        for by in range(2, max_bins+1):
            hist, _, _ = np.histogram2d(x, y, bins=(bx, by), range=((0,1),(0,1)))
            norm = np.log(min(bx, by))
            if norm > 0: best = max(best, _mi_from_hist(hist)/norm)
    return float(np.clip(best, 0.0, 1.0))


def approximate_mic_matrix(values: np.ndarray, mask: np.ndarray | None = None,
                           max_bins: int = 8) -> np.ndarray:
    values = np.asarray(values, float)
    if values.ndim != 2: raise ValueError("values must be [T, M]")
    if mask is None: mask = np.isfinite(values)
    m = values.shape[1]; out = np.eye(m, dtype=np.float32)
    for i in range(m):
        for j in range(i+1, m):
            valid = (mask[:,i] > 0) & (mask[:,j] > 0)
            score = approximate_mic(values[valid,i], values[valid,j], max_bins=max_bins)
            out[i,j] = out[j,i] = score
    return out


def build_candidate_graph(mic: np.ndarray, top_k: int = 5, threshold: float = 0.1,
                          self_loops: bool = True) -> np.ndarray:
    mic = np.asarray(mic, dtype=np.float32)
    if mic.ndim != 2 or mic.shape[0] != mic.shape[1]: raise ValueError("mic must be square")
    m = mic.shape[0]; directed = np.zeros_like(mic)
    for i in range(m):
        order = np.argsort(mic[i])[::-1]
        keep = [j for j in order if j != i and mic[i,j] >= threshold][:top_k]
        directed[i,keep] = mic[i,keep]
    graph = np.maximum(directed, directed.T)  # union-rule symmetrization
    if self_loops: np.fill_diagonal(graph, 1.0)
    return graph
