from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from typing import Any

import numpy as np


MISSING_PATTERN_IDS = {
    "none": 0,
    "isolated": 1,
    "short_burst": 2,
    "long_gap": 3,
    "correlated_channel": 4,
    "drift_collision": 5,
}
DRIFT_TYPE_IDS = {
    "none": 0,
    "jitter": 1,
    "heavy_tailed": 2,
    "accumulated": 3,
    "reset": 4,
}


@dataclass
class CorruptionConfig:
    """Configuration for the reality-informed degradation protocol.

    ``affected_ratio`` is an occurrence ratio over originally valid channel-time
    positions. Affected positions are split into missing observations and timestamp
    displacement by ``drift_fraction``. Timestamp offsets are generated in units of
    nominal sampling intervals and converted to the same physical-time unit as
    ``reference_times`` before they are returned.
    """

    affected_ratio: float = 0.05
    drift_fraction: float = 0.40

    # Missingness. ``ratio_missing_pattern_weights`` may override the base mixture
    # for the four affected-data ratios used in the paper.
    missing_pattern_weights: dict[str, float] | None = None
    ratio_missing_pattern_weights: dict[str, dict[str, float]] | None = None
    short_burst_length: tuple[int, int] = (2, 8)
    long_gap_median_samples: float = 15.0
    long_gap_p95_samples: float = 90.0
    long_gap_length_bounds: tuple[int, int] = (3, 180)
    correlated_group_size: tuple[int, int] = (2, 6)

    # Timestamp displacement. The four processes correspond to the manuscript:
    # local jitter, heavy-tailed transient offsets, accumulated drift, and drift
    # that grows within a contact interval and resets at the next contact.
    drift_type_weights: dict[str, float] | None = None
    drift_sigma_intervals: float = 2.0
    max_abs_drift_intervals: float = 5.0
    min_abs_drift_intervals: float = 0.05
    heavy_tail_df: float = 3.0
    accumulated_step_sigma_intervals: float = 0.25
    accumulated_length: tuple[int, int] = (8, 48)
    reset_period_mean_samples: float = 100.0
    reset_period_bounds: tuple[int, int] = (12, 200)
    collision_to_missing: bool = False

    measurement_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if self.missing_pattern_weights is None:
            # Historical calibration: 30% isolated and 70% contiguous/structured
            # missing mass. Ratio-specific profiles below shift mass toward longer
            # gaps as the affected-data ratio increases.
            self.missing_pattern_weights = {
                "isolated": 0.30,
                "short_burst": 0.35,
                "long_gap": 0.20,
                "correlated_channel": 0.15,
            }
        if self.ratio_missing_pattern_weights is None:
            self.ratio_missing_pattern_weights = {
                "0.03": {"isolated": 0.30, "short_burst": 0.40, "long_gap": 0.15, "correlated_channel": 0.15},
                "0.05": {"isolated": 0.25, "short_burst": 0.38, "long_gap": 0.20, "correlated_channel": 0.17},
                "0.08": {"isolated": 0.18, "short_burst": 0.34, "long_gap": 0.28, "correlated_channel": 0.20},
                "0.10": {"isolated": 0.10, "short_burst": 0.25, "long_gap": 0.45, "correlated_channel": 0.20},
            }
        if self.drift_type_weights is None:
            self.drift_type_weights = {
                "jitter": 0.25,
                "heavy_tailed": 0.25,
                "accumulated": 0.25,
                "reset": 0.25,
            }
        if not 0 <= self.affected_ratio <= 1:
            raise ValueError("affected_ratio must be in [0, 1]")
        if not 0 <= self.drift_fraction <= 1:
            raise ValueError("drift_fraction must be in [0, 1]")
        if self.heavy_tail_df <= 0:
            raise ValueError("heavy_tail_df must be positive")
        if self.max_abs_drift_intervals <= 0:
            raise ValueError("max_abs_drift_intervals must be positive")

    def missing_weights_for_ratio(self) -> dict[str, float]:
        key = f"{self.affected_ratio:.2f}"
        return (self.ratio_missing_pattern_weights or {}).get(
            key, self.missing_pattern_weights or {}
        )


def resolve_corruption_config(
    raw: dict[str, Any], *, affected_ratio: float | None = None, dataset: str | None = None
) -> CorruptionConfig:
    """Resolve YAML-style config, optional dataset overrides, and ratio profiles.

    A config may contain a ``dataset_profiles`` mapping. Each selected profile is
    merged over the top-level settings before constructing ``CorruptionConfig``.
    This keeps the generator paper-auditable without hard-coding dataset names in
    the implementation.
    """

    resolved = copy.deepcopy(raw)
    profiles = resolved.pop("dataset_profiles", {}) or {}
    if dataset is not None and dataset in profiles:
        for key, value in (profiles[dataset] or {}).items():
            resolved[key] = copy.deepcopy(value)
    if affected_ratio is not None:
        resolved["affected_ratio"] = float(affected_ratio)
    for key in (
        "short_burst_length",
        "long_gap_length_bounds",
        "correlated_group_size",
        "accumulated_length",
        "reset_period_bounds",
    ):
        if key in resolved:
            resolved[key] = tuple(resolved[key])
    return CorruptionConfig(**resolved)


def _choice_weighted(rng: np.random.Generator, weights: dict[str, float]) -> str:
    if not weights:
        raise ValueError("weight dictionary must not be empty")
    names = list(weights)
    probs = np.asarray([weights[n] for n in names], dtype=float)
    if np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("weights must be nonnegative and sum to a positive value")
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def _valid_candidates(natural: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    return np.argwhere((natural > 0) & (~occupied))


def _nominal_interval(reference_times: np.ndarray) -> float:
    dt = np.diff(np.asarray(reference_times, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else 1.0


def _sample_lognormal_length(rng: np.random.Generator, median: float, p95: float,
                             bounds: tuple[int, int]) -> int:
    lo, hi = int(bounds[0]), int(bounds[1])
    median = max(float(median), 1.0)
    p95 = max(float(p95), median)
    sigma = 0.0 if p95 == median else np.log(p95 / median) / 1.6448536269514722
    value = median if sigma == 0 else rng.lognormal(mean=np.log(median), sigma=sigma)
    return int(np.clip(round(value), lo, hi))


def _select_correlated_channels(rng: np.random.Generator, graph: np.ndarray | None,
                                anchor: int, size: int, m: int) -> np.ndarray:
    size = max(1, min(int(size), m))
    if graph is None:
        choices = np.arange(m)
        rng.shuffle(choices)
        return choices[:size]
    scores = np.asarray(graph[anchor], dtype=float).copy()
    scores[anchor] = np.nanmax(scores) + 1.0 if scores.size else 1.0
    return np.argsort(scores)[::-1][:size]


def _add_missing_positions(mask: np.ndarray, pattern_id: np.ndarray,
                           natural: np.ndarray, occupied: np.ndarray,
                           channels: np.ndarray, start: int, length: int,
                           pid: int, remaining: int) -> int:
    end = min(mask.shape[0], start + max(1, int(length)))
    added = 0
    for ti in range(start, end):
        for ch in channels:
            if added >= remaining:
                return added
            if natural[ti, ch] > 0 and not occupied[ti, ch]:
                mask[ti, ch] = True
                occupied[ti, ch] = True
                pattern_id[ti, ch] = pid
                added += 1
    return added


def _make_missing_mask(rng: np.random.Generator, natural: np.ndarray, target: int,
                       cfg: CorruptionConfig, graph: np.ndarray | None,
                       occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, m = natural.shape
    out = np.zeros_like(natural, dtype=bool)
    pattern_id = np.zeros_like(natural, dtype=np.int8)
    weights = cfg.missing_weights_for_ratio()
    attempts = 0

    while int(out.sum()) < target and attempts < max(2000, target * 30):
        attempts += 1
        remaining = target - int(out.sum())
        pattern = _choice_weighted(rng, weights)
        if pattern == "isolated":
            cand = _valid_candidates(natural, occupied)
            if not len(cand):
                break
            ti, ch = cand[rng.integers(len(cand))]
            out[ti, ch] = True
            occupied[ti, ch] = True
            pattern_id[ti, ch] = MISSING_PATTERN_IDS[pattern]
        elif pattern == "short_burst":
            ch = int(rng.integers(m))
            start = int(rng.integers(t))
            length = int(rng.integers(cfg.short_burst_length[0], cfg.short_burst_length[1] + 1))
            _add_missing_positions(out, pattern_id, natural, occupied, np.array([ch]),
                                   start, length, MISSING_PATTERN_IDS[pattern], remaining)
        elif pattern == "long_gap":
            ch = int(rng.integers(m))
            start = int(rng.integers(t))
            length = _sample_lognormal_length(
                rng, cfg.long_gap_median_samples, cfg.long_gap_p95_samples,
                cfg.long_gap_length_bounds,
            )
            _add_missing_positions(out, pattern_id, natural, occupied, np.array([ch]),
                                   start, length, MISSING_PATTERN_IDS[pattern], remaining)
        elif pattern == "correlated_channel":
            anchor = int(rng.integers(m))
            upper = min(m, cfg.correlated_group_size[1])
            size = int(rng.integers(cfg.correlated_group_size[0], upper + 1))
            channels = _select_correlated_channels(rng, graph, anchor, size, m)
            start = int(rng.integers(t))
            # Correlated loss is an interval shared by several related channels.
            length = int(rng.integers(cfg.short_burst_length[0], cfg.short_burst_length[1] + 1))
            _add_missing_positions(out, pattern_id, natural, occupied, channels,
                                   start, length, MISSING_PATTERN_IDS[pattern], remaining)
        else:
            raise ValueError(f"unknown missing pattern: {pattern}")

    # Fill any exact-count deficit with isolated valid points; this does not alter
    # previously created blocks and makes the requested affected ratio reproducible.
    if int(out.sum()) < target:
        cand = _valid_candidates(natural, occupied)
        n = min(target - int(out.sum()), len(cand))
        if n:
            take = cand[rng.choice(len(cand), size=n, replace=False)]
            out[take[:, 0], take[:, 1]] = True
            occupied[take[:, 0], take[:, 1]] = True
            pattern_id[take[:, 0], take[:, 1]] = MISSING_PATTERN_IDS["isolated"]
    return out, pattern_id


def _clip_nonzero(values: np.ndarray, cfg: CorruptionConfig) -> np.ndarray:
    values = np.clip(values, -cfg.max_abs_drift_intervals, cfg.max_abs_drift_intervals)
    small = np.abs(values) < cfg.min_abs_drift_intervals
    if np.any(small):
        signs = np.where(values[small] < 0, -1.0, 1.0)
        values[small] = signs * cfg.min_abs_drift_intervals
    return values


def _truncated_gaussian(rng: np.random.Generator, n: int, cfg: CorruptionConfig) -> np.ndarray:
    vals = rng.normal(0.0, cfg.drift_sigma_intervals, n)
    return _clip_nonzero(vals, cfg)


def _add_drift_single(rng: np.random.Generator, natural: np.ndarray, occupied: np.ndarray,
                      mask: np.ndarray, type_id: np.ndarray, offsets_intervals: np.ndarray,
                      kind: str, cfg: CorruptionConfig) -> int:
    cand = _valid_candidates(natural, occupied)
    if not len(cand):
        return 0
    ti, ch = cand[rng.integers(len(cand))]
    if kind == "jitter":
        value = _truncated_gaussian(rng, 1, cfg)[0]
    elif kind == "heavy_tailed":
        value = rng.standard_t(df=cfg.heavy_tail_df) * cfg.drift_sigma_intervals / np.sqrt(cfg.heavy_tail_df)
        value = _clip_nonzero(np.asarray([value], dtype=float), cfg)[0]
    else:
        raise ValueError(kind)
    mask[ti, ch] = True
    occupied[ti, ch] = True
    type_id[ti, ch] = DRIFT_TYPE_IDS[kind]
    offsets_intervals[ti, ch] = value
    return 1


def _available_interval(natural: np.ndarray, occupied: np.ndarray, ch: int,
                        start: int, length: int, remaining: int) -> np.ndarray:
    end = min(natural.shape[0], start + max(1, int(length)))
    ids = np.arange(start, end)
    ids = ids[(natural[ids, ch] > 0) & (~occupied[ids, ch])]
    return ids[:remaining]


def _make_drift(rng: np.random.Generator, natural: np.ndarray, target: int,
                cfg: CorruptionConfig, occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, m = natural.shape
    mask = np.zeros_like(natural, dtype=bool)
    type_id = np.zeros_like(natural, dtype=np.int8)
    offsets_intervals = np.zeros_like(natural, dtype=np.float32)
    attempts = 0

    while int(mask.sum()) < target and attempts < max(2000, target * 40):
        attempts += 1
        remaining = target - int(mask.sum())
        kind = _choice_weighted(rng, cfg.drift_type_weights or {})
        if kind in {"jitter", "heavy_tailed"}:
            _add_drift_single(rng, natural, occupied, mask, type_id,
                              offsets_intervals, kind, cfg)
            continue

        ch = int(rng.integers(m))
        start = int(rng.integers(t))
        if kind == "accumulated":
            length = int(rng.integers(cfg.accumulated_length[0], cfg.accumulated_length[1] + 1))
            ids = _available_interval(natural, occupied, ch, start, length, remaining)
            if not len(ids):
                continue
            steps = rng.normal(0.0, cfg.accumulated_step_sigma_intervals, len(ids))
            vals = _clip_nonzero(np.cumsum(steps), cfg)
        elif kind == "reset":
            lo, hi = cfg.reset_period_bounds
            sampled = int(round(rng.exponential(cfg.reset_period_mean_samples)))
            length = int(np.clip(sampled, lo, hi))
            ids = _available_interval(natural, occupied, ch, start, length, remaining)
            if not len(ids):
                continue
            endpoint = _truncated_gaussian(rng, 1, cfg)[0]
            vals = np.linspace(endpoint / len(ids), endpoint, len(ids), dtype=float)
            vals = _clip_nonzero(vals, cfg)
        else:
            raise ValueError(f"unknown drift type: {kind}")

        mask[ids, ch] = True
        occupied[ids, ch] = True
        type_id[ids, ch] = DRIFT_TYPE_IDS[kind]
        offsets_intervals[ids, ch] = vals.astype(np.float32)

    # Exact-count fallback: isolated jitter positions.
    while int(mask.sum()) < target:
        if not _add_drift_single(rng, natural, occupied, mask, type_id,
                                 offsets_intervals, "jitter", cfg):
            break
    return mask, type_id, offsets_intervals


def _convert_collisions_to_missing(reference_times: np.ndarray, natural: np.ndarray,
                                   missing: np.ndarray, missing_id: np.ndarray,
                                   drift: np.ndarray, drift_id: np.ndarray,
                                   offsets_seconds: np.ndarray, nominal_dt: float) -> int:
    """Convert displaced events that collide after nearest-grid quantization to missing.

    Bi-NSDE consumes continuous event times, so no resampling is otherwise performed.
    Collision handling only prevents two received records from occupying the same nominal
    reception bin when a displaced event crosses another sample.
    """

    t, m = natural.shape
    converted = 0
    for ch in range(m):
        occupied_bins: set[int] = set()
        # Undisplaced, received natural observations reserve their original bins.
        for ti in range(t):
            if natural[ti, ch] <= 0 or missing[ti, ch] or drift[ti, ch]:
                continue
            occupied_bins.add(ti)
        for ti in np.where(drift[:, ch])[0]:
            shifted = reference_times[ti] + float(offsets_seconds[ti, ch])
            target_bin = int(round((shifted - reference_times[0]) / nominal_dt)) if nominal_dt > 0 else ti
            if target_bin < 0 or target_bin >= t or target_bin in occupied_bins:
                drift[ti, ch] = False
                drift_id[ti, ch] = DRIFT_TYPE_IDS["none"]
                offsets_seconds[ti, ch] = 0.0
                missing[ti, ch] = True
                missing_id[ti, ch] = MISSING_PATTERN_IDS["drift_collision"]
                converted += 1
            else:
                occupied_bins.add(target_bin)
    return converted


def _counts_by_id(ids: np.ndarray, mapping: dict[str, int]) -> dict[str, int]:
    return {name: int(np.sum(ids == value)) for name, value in mapping.items() if value != 0}


def apply_corruption(values: np.ndarray, reference_times: np.ndarray, natural_mask: np.ndarray,
                     cfg: CorruptionConfig, seed: int,
                     graph: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Apply controlled missingness and timestamp displacement.

    Missing observation:
      * the original reference value remains in ``values`` only as the target;
      * ``received_values`` is zeroed at that position;
      * ``observation_mask`` is set to zero.

    Timestamp displacement:
      * the value is kept in ``received_values``;
      * ``observation_mask`` remains one;
      * the event time becomes ``reference_time + timestamp_offset``.

    The two sampled sets are disjoint and only originally valid positions can be affected.
    """

    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float32)
    reference_times = np.asarray(reference_times, dtype=np.float64)
    natural_mask = np.asarray(natural_mask, dtype=np.float32)
    if values.ndim != 2 or natural_mask.shape != values.shape:
        raise ValueError("values and natural_mask must both have shape [T, M]")
    if reference_times.shape != (values.shape[0],):
        raise ValueError("reference_times must have shape [T]")

    valid_count = int(natural_mask.sum())
    affected_target = min(valid_count, int(round(cfg.affected_ratio * valid_count)))
    drift_target = int(round(cfg.drift_fraction * affected_target))
    missing_target = affected_target - drift_target

    occupied = np.zeros_like(natural_mask, dtype=bool)
    missing, missing_id = _make_missing_mask(
        rng, natural_mask, missing_target, cfg, graph, occupied
    )
    drift, drift_id, offsets_intervals = _make_drift(
        rng, natural_mask, drift_target, cfg, occupied
    )

    nominal_dt = _nominal_interval(reference_times)
    offsets_seconds = offsets_intervals * nominal_dt
    collision_count = 0
    if cfg.collision_to_missing and drift.any():
        collision_count = _convert_collisions_to_missing(
            reference_times, natural_mask, missing, missing_id,
            drift, drift_id, offsets_seconds, nominal_dt,
        )

    received = values.copy()
    if cfg.measurement_noise_std > 0:
        received += (
            rng.normal(0, cfg.measurement_noise_std, values.shape).astype(np.float32)
            * natural_mask
        )
    received[missing] = 0.0

    observation = natural_mask.copy()
    observation[missing] = 0.0
    evaluation = ((missing | drift) & (natural_mask > 0)).astype(np.float32)
    observation_times = reference_times[:, None] + offsets_seconds

    missing_count = int(missing.sum())
    drift_count = int(drift.sum())
    realized_missing = float(missing_count / max(valid_count, 1))
    realized_drift = float(drift_count / max(valid_count, 1))
    meta = {
        "seed": seed,
        "config": asdict(cfg),
        "nominal_interval": nominal_dt,
        "timestamp_offset_unit": "same unit as reference_times",
        "valid_count": valid_count,
        "requested_affected_count": affected_target,
        "missing_count": missing_count,
        "drift_count": drift_count,
        "collision_to_missing_count": collision_count,
        "realized_missing_ratio": realized_missing,
        "realized_drift_ratio": realized_drift,
        "realized_affected_ratio": realized_missing + realized_drift,
        "missing_pattern_counts": _counts_by_id(missing_id, MISSING_PATTERN_IDS),
        "drift_type_counts": _counts_by_id(drift_id, DRIFT_TYPE_IDS),
    }

    return {
        "values": values,
        "reference_times": reference_times,
        "natural_mask": natural_mask,
        "received_values": received.astype(np.float32),
        "observation_mask": observation.astype(np.float32),
        "observation_times": observation_times.astype(np.float64),
        "timestamp_offsets": offsets_seconds.astype(np.float32),
        "artificial_missing_mask": missing.astype(np.float32),
        "artificial_drift_mask": drift.astype(np.float32),
        "missing_pattern_id": missing_id,
        "drift_type_id": drift_id,
        "evaluation_mask": evaluation,
        "metadata_json": np.asarray(json.dumps(meta)),
    }
