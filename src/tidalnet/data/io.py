from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from .schema import TelemetryArrays, validate_corrupted


def load_npz(path: str | Path, validate: bool = True) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    if validate:
        TelemetryArrays(data["values"], data["reference_times"], data["natural_mask"]).validate()
        if "received_values" in data:
            validate_corrupted(data)
    return data


def save_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
