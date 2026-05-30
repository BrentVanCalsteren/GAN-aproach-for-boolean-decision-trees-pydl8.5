import numpy as np
from src.samplers.sampler import Sampler

class SingleGaussian1D_sampler:
    mean = None
    var = None
    log_norm_const = None

    def fit(self, points):
        points = np.asarray(points, dtype=np.float64).ravel()
        self.mean = points.mean()
        self.var = points.var()
        if self.var < 1e-6:
            self.var = 1e-6
        self.log_norm_const = 0.5 * np.log(2 * np.pi * self.var)

    def score_feature(self, feature:np.ndarray) -> np.ndarray:#can be array or single val
        feature_arr = np.asarray(feature)
        # Standard log-pdf formula
        return -0.5 * ((feature_arr - self.mean) ** 2 / self.var) - self.log_norm_const

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.RandomState(random_state)
        #point = np.array([-1])
        #while point[point < 0].shape[0] > 0 or point[point > 1].shape[0] > 0:

        point = self.mean + rng.randn(n_samples) * np.sqrt(self.var)
        return point

    def sorted_samples(self, n: int) -> np.ndarray:
        candidates = self.sample(n_samples=n)
        ll = self.score_feature(candidates)
        return candidates[np.argsort(-ll)]

    def calc_norm_conf_each_sample(self, distributions, samples):
        features = samples.T

        n_samples = samples.shape[0]
        total_log_likelihoods = np.zeros(n_samples)
        total_max_log_likelihood = 0.0

        for i, gm in enumerate(distributions):
            total_log_likelihoods += gm.score_feature(features[i])
            total_max_log_likelihood += gm.max_log_likelihood

        return np.exp(total_log_likelihoods - total_max_log_likelihood)