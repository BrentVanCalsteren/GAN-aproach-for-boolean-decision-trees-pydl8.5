import numpy as np


class Bernoulli_Distr:
    prob = None
    log_prob = None
    log_prob_neg = None

    def __init__(self):
        return

    def fit(self, points):
        points = np.asarray(points, dtype=np.float64).ravel()
        self.prob = (points.sum() + 1e-6) / (len(points) + 2e-6)#Laplace smoothing
        self.log_prob = np.log(self.prob)
        self.log_prob_neg = np.log(1.0 - self.prob)

    def score_feature(self, feature):
        feature = np.asarray(feature, dtype=np.float64).ravel()
        return feature * self.log_prob + (1.0 - feature) * self.log_prob_neg

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        return (rng.rand(n_samples) < self.prob).astype(np.float64)