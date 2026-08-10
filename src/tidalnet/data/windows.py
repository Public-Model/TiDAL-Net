from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class TelemetryWindowDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray], start: int, end: int,
                 window: int, stride: int = 1):
        self.data = data; self.start = start; self.end = end
        self.window = window; self.stride = stride
        self.indices = list(range(start, max(start, end-window+1), stride))
        if not self.indices and end-start >= window: self.indices = [start]

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = self.indices[index]; e = s + self.window
        times = self.data["reference_times"][s:e]
        sample = {
            "target": self.data["values"][s:e],
            "received": self.data.get("received_values", self.data["values"])[s:e],
            "mask": self.data.get("observation_mask", self.data["natural_mask"])[s:e],
            "natural_mask": self.data["natural_mask"][s:e],
            "offsets": self.data.get("timestamp_offsets", np.zeros_like(self.data["values"]))[s:e],
            "eval_mask": self.data.get("evaluation_mask", self.data["natural_mask"])[s:e],
            "times": times,
        }
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in sample.items()}


def chronological_bounds(length: int, split: list[float] | tuple[float,float,float]) -> dict[str, tuple[int,int]]:
    if len(split) != 3 or abs(sum(split)-1.0) > 1e-6:
        raise ValueError("split must contain three fractions summing to one")
    a = int(length*split[0]); b = int(length*(split[0]+split[1]))
    return {"train": (0,a), "val": (a,b), "test": (b,length)}
