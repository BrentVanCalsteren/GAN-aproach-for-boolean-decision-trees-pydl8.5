import numpy as np


class Multinomial_distr:
    probs = None       #probability for each category
    log_probs = None   #log probabilities for scoring
    n_categories = None

    def __init__(self):
        return

    def fit(self, points):
        points = np.asarray(points, dtype=np.int64).ravel()
        self.n_categories = int(points.max()) + 1

        counts = np.bincount(points, minlength=self.n_categories).astype(np.float64)

        counts += 1e-6 #Laplace smoothing
        self.probs = counts / counts.sum()
        self.log_probs = np.log(self.probs)

    def score_feature(self, feature):
        feature = np.asarray(feature, dtype=np.int64).ravel()
        return self.log_probs[feature]

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        return rng.choice(self.n_categories, size=n_samples, p=self.probs)