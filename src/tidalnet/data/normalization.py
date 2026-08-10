from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ZScoreNormalizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    epsilon: float = 1e-6

    def fit(self, values: np.ndarray, mask: np.ndarray) -> "ZScoreNormalizer":
        masked = np.where(mask > 0, values, np.nan)
        self.mean = np.nanmean(masked, axis=0)
        self.std = np.nanstd(masked, axis=0)
        self.mean = np.nan_to_num(self.mean, nan=0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std > self.epsilon), self.std, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("normalizer is not fitted")
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("normalizer is not fitted")
        return values * self.std + self.mean
