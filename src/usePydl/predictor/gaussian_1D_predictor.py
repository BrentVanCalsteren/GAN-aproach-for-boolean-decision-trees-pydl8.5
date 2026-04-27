from typing import List
from src.usePydl.predictor.predictor_obj import Predictor
from src.usePydl.error_fun import gaussian_error
from src.usePydl.leaf import leaf_val, get_leaf_vals
from src.distributions.single_gaussian import SingleGaussian1D_distr
import numpy as np

class Gaussian1DPredictor(Predictor):
    #Fit Complexity: O(n·d)
    #score complexity: 	O(n·d)
    def __init__(self,samples, samples_bin, max_depth=3,min_sup=1,time=100):
        self.samples = samples
        super().__init__(
            samples_bin=samples_bin,
            error_fun=gaussian_error(self, samples),
            leaf_val=leaf_val(self,samples),
            max_depth=max_depth,
            min_sup=min_sup,
            time=time
        )
        self.generate_tree()

    def get_distr(self, feature_array):
        gm = SingleGaussian1D_distr()
        gm.fit(feature_array.reshape(-1, 1))
        return gm


    def calc_norm_conf_each_sample(self, distributions: List[SingleGaussian1D_distr], samples):
        features = samples.T  # (n_features, n_samples)
        log_likelihoods = np.zeros(samples.shape[0])
        log_likelihoods_max = np.zeros(samples.shape[0])  # array of feature length
        for i, gm in enumerate(distributions):
            # Log-likelihood for all samples
            log_like = gm.score_feature(features[i].reshape(-1, 1))  # [ll1 ll2 ll3 ...]
            # returns prob dens vals: pdf(x) = 1 / (σ * √(2π)) * exp( - (x - μ)² / (2σ²) )
            # density of 2.5 as "2.5 times more confident"—it reflects sharper distribution.
            log_likelihoods += log_like

            mu = gm.mean  # get max log-likelihood (at the mean) for normalizing to value [0,1]
            ll_max = gm.score_feature([[mu]])
            log_likelihoods_max += ll_max  # is 1 value but np will map over entire array
            # Normalise
        confidence = np.exp(log_likelihoods - log_likelihoods_max)
        return confidence

    def _generate_new_leaf_samples(self, n, distributions: List[SingleGaussian1D_distr], conf_tresh):
        samples_above_tresh = []
        while len(samples_above_tresh) < n:
            features = []
            for distr in distributions:
                gen = distr.sample(n_samples=100)  # returns ([[10*[f1]],[10*[f2]],...] , Label)
                features.append(gen)
            features = np.array(features)
            good_features = []
            z_prob_matrix = calc_norm_conf_sample_x_feature(distributions, features.T)
            for i, feature in enumerate(z_prob_matrix.T):
                indices = np.argsort(feature)
                good_points = features[i, indices]
                good_features.append(good_points)
            samples = np.array(good_features).T
            sample_prob = self.calc_norm_conf_each_sample(distributions, samples)
            for i, prob in enumerate(sample_prob):
                if prob >= conf_tresh:
                    samples_above_tresh.append(samples[i])
        return np.array(samples_above_tresh)[:n]





#GENERAL FUNCTIONS - prob estimation fucntions





def calc_norm_conf_sample_x_feature(distributions: List[SingleGaussian1D_distr], samples):
    n_samples, n_features = samples.shape
    features = samples.T  # shape (n_features, n_samples)
    #compute max log‑likelihood each feature (=mean)
    max_ll_per_feature = np.zeros(n_features)
    for i, gm in enumerate(distributions):
        mu = gm.mean
        max_ll_per_feature[i] = gm.score_feature([[mu]])
    #FEATxSAMPLE matrix logs
    log_likelihoods = np.zeros((n_features, n_samples))
    for i, gm in enumerate(distributions):
        log_likelihoods[i] = gm.score_feature(features[i].reshape(-1, 1))
    #normilize
    confidence = np.exp(log_likelihoods - max_ll_per_feature[:, np.newaxis])
    return confidence.T  #shape (n_samples, n_features)
