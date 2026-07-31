from typing import Dict, List

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
    def generate_new_samples_for_all_features_of_this_type(cls, indices, gen_feats_matrix, conf_thresh: float, samplers: list, intervals_list: List = None):
        sampler = samplers[0]
        n = gen_feats_matrix.shape[1]
        good_samples = np.array([])
        attempts = 0
        while (good_samples.size == 0 or good_samples.shape[0] < n) and attempts < 100:
            sub_samples = sampler.sorted_samples(n=n)
            scores = sampler.score_feature(sub_samples)
            valid_mask = scores >= conf_thresh

            if intervals_list:
                for i in range(sub_samples.shape[1]):
                    feat_intervals = intervals_list[i]
                    feat_vals = sub_samples[:, i]
                    feat_valid = np.zeros_like(feat_vals, dtype=bool)
                    for inter in feat_intervals:
                        feat_valid |= (feat_vals >= inter[0]) & (feat_vals <= inter[1])
                    valid_mask &= feat_valid

            valid_samples = sub_samples[valid_mask]
            if good_samples.shape[0] > 0:
                good_samples = np.vstack((good_samples, valid_samples))
            else:
                good_samples = valid_samples
            attempts += 1

        if good_samples.shape[0] < n:
            needed = n - good_samples.shape[0]
            fallback = sampler.sample(needed)
            if fallback.ndim == 1:
                fallback = fallback.reshape(1, -1)
            if intervals_list:
                for i in range(fallback.shape[1]):
                    fallback[:, i] = np.clip(fallback[:, i], intervals_list[i][0][0], intervals_list[i][0][1])
            if good_samples.shape[0] > 0:
                good_samples = np.vstack((good_samples, fallback))
            else:
                good_samples = fallback

        good_feats = good_samples[:n, :].T
        j = 0
        for i in range(gen_feats_matrix.shape[0]):
            if indices[i]:
                gen_feats_matrix[i] = good_feats[j]
                j += 1
