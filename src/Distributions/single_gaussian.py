import numpy as np

class SingleGaussian1D_distr:
    mean = None
    var = None
    log_scale = None

    def __init__(self):
        return

    def fit(self, points):
        points = np.asarray(points, dtype=np.float64).ravel()
        self.mean = points.mean()
        self.var = points.var()
        if self.var < 1e-6:
            self.var = 1e-6
        self.log_scale = 0.5 * np.log(2 * np.pi * self.var)

    def score_feature(self, feature):
        feature = np.asarray(feature).ravel()
        return -0.5 * ((feature - self.mean)**2 / self.var) - self.log_scale

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        return self.mean + rng.randn(n_samples) * np.sqrt(self.var)