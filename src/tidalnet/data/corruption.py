from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import numpy as np


@dataclass
class CorruptionConfig:
    affected_ratio: float = 0.05
    drift_fraction: float = 0.35
    missing_pattern_weights: dict[str, float] | None = None
    short_burst_length: tuple[int, int] = (2, 8)
    long_gap_length: tuple[int, int] = (12, 48)
    correlated_group_size: tuple[int, int] = (2, 6)
    drift_type_weights: dict[str, float] | None = None
    drift_scale: float = 0.35
    max_abs_drift: float = 2.0
    measurement_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if self.missing_pattern_weights is None:
            self.missing_pattern_weights = {
                "isolated": 0.30, "short_burst": 0.30,
                "long_gap": 0.20, "correlated_channel": 0.20,
            }
        if self.drift_type_weights is None:
            self.drift_type_weights = {
                "gaussian": 0.20, "uniform": 0.20, "heavy_tailed": 0.20,
                "accumulated": 0.20, "reset": 0.20,
            }
        if not 0 <= self.affected_ratio <= 1:
            raise ValueError("affected_ratio must be in [0, 1]")
        if not 0 <= self.drift_fraction <= 1:
            raise ValueError("drift_fraction must be in [0, 1]")


def _choice_weighted(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    probs = np.asarray([weights[n] for n in names], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def _valid_candidates(natural: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    return np.argwhere((natural > 0) & (~occupied))


def _add_interval(mask: np.ndarray, natural: np.ndarray, occupied: np.ndarray,
                  channels: np.ndarray, start: int, length: int) -> int:
    end = min(mask.shape[0], start + max(1, length))
    before = int(mask.sum())
    for ch in channels:
        valid = (natural[start:end, ch] > 0) & (~occupied[start:end, ch])
        mask[start:end, ch][valid] = True
        occupied[start:end, ch][valid] = True
    return int(mask.sum()) - before


def _select_correlated_channels(rng: np.random.Generator, graph: np.ndarray | None,
                                anchor: int, size: int, m: int) -> np.ndarray:
    if graph is None:
        choices = np.arange(m)
        rng.shuffle(choices)
        return choices[:size]
    scores = np.asarray(graph[anchor]).copy()
    scores[anchor] = scores.max(initial=1.0) + 1.0
    return np.argsort(scores)[::-1][:size]


def _make_missing_mask(rng: np.random.Generator, natural: np.ndarray, target: int,
                       cfg: CorruptionConfig, graph: np.ndarray | None,
                       occupied: np.ndarray) -> np.ndarray:
    t, m = natural.shape
    out = np.zeros_like(natural, dtype=bool)
    attempts = 0
    while out.sum() < target and attempts < max(1000, target * 20):
        attempts += 1
        pattern = _choice_weighted(rng, cfg.missing_pattern_weights or {})
        if pattern == "isolated":
            cand = _valid_candidates(natural, occupied)
            if not len(cand): break
            ti, ch = cand[rng.integers(len(cand))]
            out[ti, ch] = True; occupied[ti, ch] = True
        elif pattern == "short_burst":
            ch = int(rng.integers(m)); start = int(rng.integers(t))
            length = int(rng.integers(cfg.short_burst_length[0], cfg.short_burst_length[1] + 1))
            _add_interval(out, natural, occupied, np.array([ch]), start, length)
        elif pattern == "long_gap":
            ch = int(rng.integers(m)); start = int(rng.integers(t))
            length = int(rng.integers(cfg.long_gap_length[0], cfg.long_gap_length[1] + 1))
            _add_interval(out, natural, occupied, np.array([ch]), start, length)
        else:
            anchor = int(rng.integers(m))
            size = int(rng.integers(cfg.correlated_group_size[0], min(m, cfg.correlated_group_size[1]) + 1))
            channels = _select_correlated_channels(rng, graph, anchor, size, m)
            start = int(rng.integers(t))
            length = int(rng.integers(cfg.short_burst_length[0], cfg.long_gap_length[1] + 1))
            _add_interval(out, natural, occupied, channels, start, length)
    # Trim to an exact target, then fill any deficit with isolated valid positions.
    idx = np.argwhere(out)
    if len(idx) > target:
        drop = idx[rng.choice(len(idx), size=len(idx)-target, replace=False)]
        out[drop[:,0], drop[:,1]] = False
        occupied[drop[:,0], drop[:,1]] = False
    if out.sum() < target:
        cand = _valid_candidates(natural, occupied)
        n = min(target - int(out.sum()), len(cand))
        if n:
            take = cand[rng.choice(len(cand), size=n, replace=False)]
            out[take[:,0], take[:,1]] = True
            occupied[take[:,0], take[:,1]] = True
    return out


def _drift_values(rng: np.random.Generator, kind: str, positions: np.ndarray,
                  shape: tuple[int, int], cfg: CorruptionConfig) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    n = len(positions)
    if n == 0: return out
    s = cfg.drift_scale
    if kind == "gaussian": vals = rng.normal(0.0, s, n)
    elif kind == "uniform": vals = rng.uniform(-np.sqrt(3)*s, np.sqrt(3)*s, n)
    elif kind == "heavy_tailed": vals = rng.standard_t(df=3, size=n) * s / np.sqrt(3)
    elif kind == "accumulated":
        vals = np.zeros(n)
        for ch in np.unique(positions[:,1]):
            ids = np.where(positions[:,1] == ch)[0]
            order = ids[np.argsort(positions[ids,0])]
            vals[order] = np.cumsum(rng.normal(0.0, s/3, len(order)))
    elif kind == "reset":
        vals = np.zeros(n)
        for ch in np.unique(positions[:,1]):
            ids = np.where(positions[:,1] == ch)[0]
            vals[ids] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5*s, 2*s)
    else: raise ValueError(f"unknown drift type: {kind}")
    vals = np.clip(vals, -cfg.max_abs_drift, cfg.max_abs_drift)
    out[positions[:,0], positions[:,1]] = vals.astype(np.float32)
    return out


def apply_corruption(values: np.ndarray, reference_times: np.ndarray, natural_mask: np.ndarray,
                     cfg: CorruptionConfig, seed: int, graph: np.ndarray | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float32)
    reference_times = np.asarray(reference_times, dtype=np.float64)
    natural_mask = np.asarray(natural_mask, dtype=np.float32)
    valid_count = int(natural_mask.sum())
    affected_target = int(round(cfg.affected_ratio * valid_count))
    drift_target = int(round(cfg.drift_fraction * affected_target))
    missing_target = affected_target - drift_target
    occupied = np.zeros_like(natural_mask, dtype=bool)
    missing = _make_missing_mask(rng, natural_mask, missing_target, cfg, graph, occupied)
    candidates = _valid_candidates(natural_mask, occupied)
    n_drift = min(drift_target, len(candidates))
    drift_pos = candidates[rng.choice(len(candidates), size=n_drift, replace=False)] if n_drift else np.empty((0,2), int)
    drift = np.zeros_like(natural_mask, dtype=bool)
    if n_drift: drift[drift_pos[:,0], drift_pos[:,1]] = True

    offsets = np.zeros_like(values, dtype=np.float32)
    if n_drift:
        # Assign each position a type while retaining temporal structure for accumulated/reset operators.
        labels = np.array([_choice_weighted(rng, cfg.drift_type_weights or {}) for _ in range(n_drift)])
        for kind in set(labels.tolist()):
            pos = drift_pos[labels == kind]
            offsets += _drift_values(rng, kind, pos, values.shape, cfg)

    received = values.copy()
    if cfg.measurement_noise_std > 0:
        received += rng.normal(0, cfg.measurement_noise_std, values.shape).astype(np.float32) * natural_mask
    received[missing] = 0.0
    observation = natural_mask.copy()
    observation[missing] = 0.0
    evaluation = ((missing | drift) & (natural_mask > 0)).astype(np.float32)
    realized_missing = float(missing.sum() / max(valid_count, 1))
    realized_drift = float(drift.sum() / max(valid_count, 1))
    meta = {
        "seed": seed, "config": asdict(cfg), "valid_count": valid_count,
        "missing_count": int(missing.sum()), "drift_count": int(drift.sum()),
        "realized_missing_ratio": realized_missing,
        "realized_drift_ratio": realized_drift,
        "realized_affected_ratio": realized_missing + realized_drift,
    }
    return {
        "values": values, "reference_times": reference_times, "natural_mask": natural_mask,
        "received_values": received.astype(np.float32), "observation_mask": observation.astype(np.float32),
        "timestamp_offsets": offsets, "artificial_missing_mask": missing.astype(np.float32),
        "artificial_drift_mask": drift.astype(np.float32), "evaluation_mask": evaluation,
        "metadata_json": np.asarray(json.dumps(meta)),
    }
