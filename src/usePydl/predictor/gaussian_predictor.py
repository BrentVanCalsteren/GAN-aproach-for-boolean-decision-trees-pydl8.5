from typing import List

from src.usePydl.predictor.predictor_obj import Predictor
from src.usePydl.leaf import leaf_val_gaussian_distributions
from src.usePydl.error_fun import prob_norm_error2
from sklearn.mixture import GaussianMixture
import src.chanceCalc.prob_estimator as probEstimator
import numpy as np

class GaussianPredictor(Predictor):
    def __init__(self,samples, samples_bin, max_depth=3,min_sup=2,time=100):
        self.samples = samples
        super().__init__(
            samples_bin=samples_bin,
            error_fun=prob_norm_error2(samples),
            leaf_val=leaf_val_gaussian_distributions(samples),
            max_depth=max_depth,
            min_sup=min_sup,
            time=time
        )
        self.generate_tree()

    def generate_new_data(self,n_new_samples=100,conf_tresh=0.8) -> np.ndarray:
        #todo: improve
        leafs = self.get_leaf_vals()
        samples_counts_leaf = []
        distrbution_leafs = []
        for leaf in leafs:
            distrbution_leafs.append(leaf['value']['distr'])
            samples_counts_leaf.append(leaf['value']['count'])
        new_samples = []
        total_leaf_s_count = np.sum(samples_counts_leaf)
        for i, count in enumerate(samples_counts_leaf):
            n = int((count/total_leaf_s_count) * n_new_samples)
            samples = self.get_samples_distr(n,distrbution_leafs[i],conf_tresh)
            for sample in samples:
                new_samples.append(sample)
        return np.array(new_samples)


    def get_samples_distr(self,n,distributions: List[GaussianMixture],conf_tresh):
        samples_above_tresh = []
        while len(samples_above_tresh) < n:
            features = []
            for distr in distributions:
                (gen,_) = distr.sample(n_samples=100) #returns ([[10*[f1]],[10*[f2]],...] , Label)
                feat = []
                for point in gen:
                    feat.append(point[0])
                features.append(feat)
            features = np.array(features)
            good_features = []
            z_prob_matrix = probEstimator.calc_normalised_confidence_per_feature(distributions, features.T)
            for i,feature in enumerate(z_prob_matrix.T):
                indices = np.argsort(feature)
                good_points = features[i,indices]
                good_features.append(good_points)
            samples = np.array(good_features).T
            sample_prob = probEstimator.calc_normalised_confidence_gaussian_sample(distributions, samples)
            for i,prob in enumerate(sample_prob):
                if prob >= conf_tresh:
                    samples_above_tresh.append(samples[i])
        return np.array(samples_above_tresh)


