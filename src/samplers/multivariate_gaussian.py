import numpy as np
from scipy.stats import multivariate_normal
from src.samplers.sampler import Sampler

class MultivariateGaussianSampler(Sampler):
    def __init__(self):
        self.mean = None
        self.cov = None
        self.dist = None

    def fit(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        self.mean = np.mean(points, axis=0)
        # Add a small regularizer to the diagonal to prevent singular matrix
        if points.shape[0] > 1:
            self.cov = np.cov(points, rowvar=False) + np.eye(points.shape[1]) * 1e-6
        else:
            self.cov = np.eye(points.shape[1]) * 1e-6 #(Identity Matrix)
        self.dist = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True)

    def score_feature(self, feature: np.ndarray) -> np.ndarray:
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        return self.dist.pdf(feature)

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    def sample(self, n_samples=1):
        samples = self.dist.rvs(size=n_samples)
        if n_samples == 1 and samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 1 and self.mean.shape[0] == 1:
            samples = samples.reshape(-1, 1)
        return samples

    def sorted_samples(self, n: int) -> np.ndarray:
        candidates = self.sample(n_samples=n)
        ll = self.score_feature(candidates)
        return candidates[np.argsort(-ll)]

    @classmethod
    def fit_all_features_of_this_type(cls, features: np.ndarray) -> list:
        sampler = cls()
        sampler.fit(features.T)
        return [sampler]

    @classmethod
    def generate_new_samples_for_all_features_of_this_type(cls, n: int, conf_thresh: float, samplers: list) -> np.ndarray:
        sampler = samplers[0]
        gen_feat_good = np.empty((0, sampler.mean.shape[0]))
        while gen_feat_good.shape[0] < n:
            gen_feat = sampler.sorted_samples(n=n)
            scores = sampler.score_feature(gen_feat)
            
            valid_feats = gen_feat[scores >= conf_thresh]
            if valid_feats.shape[0] > 0:
                if gen_feat_good.shape[0] > 0:
                    gen_feat_good = np.vstack((gen_feat_good, valid_feats))
                else:
                    gen_feat_good = valid_feats
        
        return gen_feat_good[:n]
