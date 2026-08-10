from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TelemetryArrays:
    values: np.ndarray
    reference_times: np.ndarray
    natural_mask: np.ndarray

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape [T, M]")
        t, m = self.values.shape
        if self.reference_times.shape != (t,):
            raise ValueError("reference_times must have shape [T]")
        if self.natural_mask.shape != (t, m):
            raise ValueError("natural_mask must have shape [T, M]")
        if np.any(np.diff(self.reference_times) < 0):
            raise ValueError("reference_times must be nondecreasing")
        if not np.isin(self.natural_mask, [0, 1]).all():
            raise ValueError("natural_mask must be binary")


def validate_corrupted(data: dict[str, np.ndarray]) -> None:
    required = ["values", "reference_times", "natural_mask", "received_values",
                "observation_mask", "timestamp_offsets", "artificial_missing_mask",
                "artificial_drift_mask", "evaluation_mask"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing arrays: {missing}")
    shape = data["values"].shape
    for key in required[2:]:
        if data[key].shape != shape:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {shape}")
    overlap = (data["artificial_missing_mask"] > 0) & (data["artificial_drift_mask"] > 0)
    if overlap.any():
        raise ValueError("artificial missing and drift masks must be disjoint")
    expected_eval = ((data["artificial_missing_mask"] + data["artificial_drift_mask"]) > 0)
    expected_eval &= data["natural_mask"] > 0
    if not np.array_equal(expected_eval.astype(np.float32), data["evaluation_mask"].astype(np.float32)):
        raise ValueError("evaluation_mask is inconsistent with artificial affected positions")
