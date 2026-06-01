import random

import numpy as np
from src.samplers.sampler import Sampler
from scipy import stats

class SingleGaussian1DSampler(Sampler):
    mean = None
    var = None
    log_norm_const = None
    best_relative_score = None

    def fit(self, points):
        points = np.asarray(points, dtype=np.float64).ravel()
        self.mean = points.mean()
        self.var = points.var()

    def score_feature(self, feature: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var)
        if std < 1e-10:
            std = 1e-10
        z_score = (feature - self.mean) / std
        score = np.exp(-0.5 * (z_score ** 2))
        return score

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    def sample(self, n_samples=1):
        std = np.sqrt(self.var)
        return np.random.normal(loc=self.mean, scale=std, size=n_samples)

    def sorted_samples(self, n: int) -> np.ndarray:
        candidates = self.sample(n_samples=n)
        ll = self.score_feature(candidates)
        return candidates[np.argsort(-ll)]