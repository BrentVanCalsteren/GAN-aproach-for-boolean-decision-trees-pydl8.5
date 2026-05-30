import numpy as np
from src.samplers.sampler import Sampler

class Multinomial_sampler:
    probs = None
    log_probs = None
    n_categories = None
    categories_ = None  # fix: was `categories`

    def fit(self, points):
        points = np.asarray(points, dtype=np.float64).ravel()
        self.categories_ = np.unique(points)  # sorted unique values
        self.n_categories = len(self.categories_)

        # Map each value to its index
        indices = np.searchsorted(self.categories_, points)
        counts = np.bincount(indices, minlength=self.n_categories).astype(np.float64)

        counts += 1e-6
        self.probs = counts / counts.sum()
        self.log_probs = np.log(self.probs)

    def score_feature(self, feature):
        if self.log_probs is None:
            raise RuntimeError("Call fit() before scoring.")
        feature = np.asarray(feature, dtype=np.float64)
        indices = np.searchsorted(self.categories_, feature)
        return self.log_probs[indices]

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        indices = rng.choice(self.n_categories, size=n_samples, p=self.probs)
        return self.categories_[indices]

    def sorted_samples(self, n: int) -> np.ndarray:
        candidates = self.sample(n_samples=n)
        ll = self.score_feature(candidates)
        return candidates[np.argsort(-ll)]
