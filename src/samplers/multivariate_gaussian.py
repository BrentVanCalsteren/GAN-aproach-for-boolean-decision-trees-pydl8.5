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

        raw_pdf = self.dist.pdf(feature)
        max_pdf = self.dist.pdf(self.dist.mean)
        scaled_score = raw_pdf / max_pdf
        return scaled_score

    def score_avg(self, feature):
        return np.mean(self.score_feature(feature))

    #this takes very long to generate good samples when there are a lot of features
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
    def generate_new_samples_for_all_features_of_this_type(cls, indices,gen_feats_matrix, conf_thresh: float, samplers: list):
        sampler = samplers[0]
        n = gen_feats_matrix.shape[1]
        good_samples = []
        while len(good_samples) < n:
            sub_samples = sampler.sorted_samples(n=n).T#returns sample dim (n_samples, n_feat_trained_on)
            scores = sampler.score_feature(sub_samples).T
            valid_sub_samples = sub_samples[scores >= conf_thresh]
            if valid_sub_samples.shape[0] > 0:
                    for sample in valid_sub_samples:
                        good_samples.append(sample)
        gen_feats_matrix[indices] = np.array(good_samples)[:n].T
