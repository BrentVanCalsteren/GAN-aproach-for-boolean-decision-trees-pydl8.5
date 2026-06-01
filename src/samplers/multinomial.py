import numpy as np
from src.samplers.sampler import Sampler

class MultinomialSampler(Sampler):
    def __init__(self):
        self.categories = None
        self.probs = None

    def fit(self, data):
        self.categories, counts = np.unique(data, return_counts=True)
        total = np.sum(counts)
        self.probs = counts / total

    def score_feature(self, points: np.ndarray):
        score_map = dict(zip(self.categories, self.probs))
        probs = np.array([score_map.get(x, 0.0) for x in points])
        best_prob = np.max(probs)
        rel_scores = probs ** 2 - best_prob ** 2 + best_prob / best_prob
        return rel_scores

    def sample(self, n_samples=1):
        if self.categories is None:
            raise ValueError("Model must be fitted before sampling.")
        return np.random.choice(self.categories, size=n_samples, p=self.probs)

    def sorted_samples(self, n):
        data = self.sample(n)
        scores = self.score_feature(data)
        # Sort indices by score in descending order
        sorted_indices = np.argsort(-scores)
        return data[sorted_indices]
