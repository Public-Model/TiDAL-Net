import json
import numpy as np

from tidalnet.data.corruption import (
    CorruptionConfig,
    apply_corruption,
    resolve_corruption_config,
)
from tidalnet.data.schema import validate_corrupted


def test_corruption_is_deterministic_disjoint_and_exact():
    x = np.arange(400, dtype=np.float32).reshape(100, 4)
    t = np.arange(100, dtype=float) * 60.0
    m = np.ones_like(x)
    cfg = CorruptionConfig(
        affected_ratio=0.10,
        drift_fraction=0.40,
        short_burst_length=(2, 4),
        long_gap_length_bounds=(5, 20),
        correlated_group_size=(2, 3),
        collision_to_missing=False,
    )
    a = apply_corruption(x, t, m, cfg, 123, np.eye(4))
    b = apply_corruption(x, t, m, cfg, 123, np.eye(4))

    assert np.array_equal(a["artificial_missing_mask"], b["artificial_missing_mask"])
    assert np.array_equal(a["timestamp_offsets"], b["timestamp_offsets"])
    assert not np.any(
        (a["artificial_missing_mask"] > 0) & (a["artificial_drift_mask"] > 0)
    )
    assert int(a["evaluation_mask"].sum()) == 40
    validate_corrupted(a)


def test_missing_and_timestamp_displacement_have_different_semantics():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    t = np.arange(200, dtype=float) * 60.0
    m = np.ones_like(x)
    cfg = CorruptionConfig(affected_ratio=0.20, drift_fraction=0.50, collision_to_missing=False)
    out = apply_corruption(x, t, m, cfg, 77, np.eye(3))

    miss = out["artificial_missing_mask"] > 0
    drift = out["artificial_drift_mask"] > 0

    assert miss.any() and drift.any()
    assert np.all(out["received_values"][miss] == 0.0)
    assert np.all(out["observation_mask"][miss] == 0.0)

    # Timestamp displacement keeps the value and availability, but changes event time.
    assert np.allclose(out["received_values"][drift], x[drift])
    assert np.all(out["observation_mask"][drift] == 1.0)
    assert np.all(np.abs(out["timestamp_offsets"][drift]) > 0.0)
    tiled_ref = np.broadcast_to(t[:, None], x.shape)
    assert np.allclose(
        out["observation_times"][drift],
        tiled_ref[drift] + out["timestamp_offsets"][drift],
    )


def test_timestamp_offsets_are_in_physical_time_units():
    x = np.ones((300, 2), dtype=np.float32)
    t = np.arange(300, dtype=float) * 60.0
    m = np.ones_like(x)
    cfg = CorruptionConfig(
        affected_ratio=0.10,
        drift_fraction=1.0,
        max_abs_drift_intervals=5.0,
        collision_to_missing=False,
    )
    out = apply_corruption(x, t, m, cfg, 99)
    drift = out["artificial_drift_mask"] > 0
    assert drift.any()
    assert np.max(np.abs(out["timestamp_offsets"][drift])) <= 5.0 * 60.0 + 1e-5
    meta = json.loads(str(out["metadata_json"]))
    assert meta["nominal_interval"] == 60.0


def test_ratio_profiles_resolve_and_shift_toward_long_gaps():
    raw = {
        "affected_ratio": 0.03,
        "ratio_missing_pattern_weights": {
            "0.03": {"isolated": 0.6, "short_burst": 0.2, "long_gap": 0.1, "correlated_channel": 0.1},
            "0.10": {"isolated": 0.1, "short_burst": 0.2, "long_gap": 0.5, "correlated_channel": 0.2},
        },
        "dataset_profiles": {"esa_adb": {"drift_fraction": 0.4}},
    }
    low = resolve_corruption_config(raw, affected_ratio=0.03, dataset="esa_adb")
    high = resolve_corruption_config(raw, affected_ratio=0.10, dataset="esa_adb")
    assert low.drift_fraction == 0.4
    assert low.missing_weights_for_ratio()["isolated"] > high.missing_weights_for_ratio()["isolated"]
    assert low.missing_weights_for_ratio()["long_gap"] < high.missing_weights_for_ratio()["long_gap"]
