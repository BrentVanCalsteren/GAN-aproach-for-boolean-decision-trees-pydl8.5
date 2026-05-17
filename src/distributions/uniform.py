import numpy as np
from pyexpat import features


class Uniform_distr:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, feat):
        self.min = np.min(feat)
        self.max = np.max(feat)

    def score_feature(self, feature) -> np.ndarray: #can be singe point or feature array -> will always return array
        feat_array = np.array(feature)
        inside = ((feat_array >= self.min) & (feat_array <= self.max)).astype(float)
        width = self.max - self.min
        if width == 1: width = 1  - 1e-10
        return inside * ((1 - width)/10+0.9) #for generation else i never get confidence above 0.9 TODO maybe fix it correctly

    def sample(self, n_samples=1):
        if self.min is None:
            raise RuntimeError("Model must be fitted before sampling.")

        return np.random.uniform(
            low=self.min,
            high=self.max,
            size=n_samples
        )

    def sorted_samples(self, n: int) -> np.ndarray:
        candidates = self.sample(n_samples=n)
        return candidates[np.argsort(-self.score_feature(candidates))]