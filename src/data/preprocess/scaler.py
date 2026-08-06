import numpy as np
from typing import Optional

class Scaler:

    def __init__(self, min_vals: Optional[np.ndarray] = None, max_vals: Optional[np.ndarray] = None):
        if min_vals is None or max_vals is None:
            self.min_vals = None
            self.max_vals = None
        elif np.asarray(min_vals).shape != np.asarray(max_vals).shape:
            print("Shape mismatch between min_vals and max_vals, initializing to None.")
            self.min_vals = None
            self.max_vals = None
        else:
            self.min_vals = np.asarray(min_vals, dtype=float)
            self.max_vals = np.asarray(max_vals, dtype=float)

    def partial_fit(self, samples: np.ndarray):
        samples_arr = np.asarray(samples, dtype=float)
        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(1, -1)

        mins = np.min(samples_arr, axis=0)
        maxs = np.max(samples_arr, axis=0)

        if self.min_vals is None or self.max_vals is None:
            self.min_vals = mins
            self.max_vals = maxs
            return self

        if samples_arr.shape[1] != self.min_vals.size:
            print(f"mismatch, samples has {samples_arr.shape[1]} features, scaler {self.min_vals.size}.")
            return self

        self.min_vals = np.minimum(self.min_vals, mins)
        self.max_vals = np.maximum(self.max_vals, maxs)
        return self

    def transform(self, samples: np.ndarray) -> np.ndarray:
        if samples is None or self.min_vals is None or self.max_vals is None:  return samples

        samples_arr = np.asarray(samples, dtype=float)
        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(1, -1)

        if self.min_vals.size != samples_arr.shape[1]:
            print(f"mismatch, expected {self.min_vals.size}, got {samples_arr.shape[1]}")
            return samples

        diffs = self.max_vals - self.min_vals
        diffs = np.where(diffs == 0, 1.0, diffs)

        scaled = (samples_arr - self.min_vals) / diffs
        return np.clip(scaled, 0.0, 1.0)

    def inverse_transform(self, samples: np.ndarray) -> np.ndarray:
        if samples is None or self.min_vals is None or self.max_vals is None:
            return samples

        samples_arr = np.asarray(samples, dtype=float)
        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(1, -1)

        samples_arr = np.clip(samples_arr, 0.0, 1.0)

        if self.min_vals.size != samples_arr.shape[1]:
            print(f"mismatch, expected {self.min_vals.size}, got {samples_arr.shape[1]}")
            return samples

        diffs = self.max_vals - self.min_vals
        origin = samples_arr * diffs + self.min_vals
        return origin