from typing import List
from src.usePydl.predictor.predictor_obj import Predictor
from src.usePydl.error_fun import predictor_error
from src.usePydl.leaf import default_leaf_val
from src.distributions.single_gaussian import SingleGaussian1D_distr
import numpy as np

class Gaussian1DPredictor(Predictor):
    #Fit Complexity: O(n·d)
    #score complexity: 	O(n·d)
    def __init__(self,samples, samples_bin, max_depth=3,min_sup=1,time=100):
        self.samples = samples
        super().__init__(
            samples_bin=samples_bin,
            error_fun=predictor_error(self, samples),
            leaf_val=default_leaf_val(self,samples),
            max_depth=max_depth,
            min_sup=min_sup,
            time=time
        )
        self.generate_tree()

    def get_distr(self, feature_array):
        gm = SingleGaussian1D_distr()
        gm.fit(feature_array.reshape(-1, 1))
        return gm

    # used in default error function
    def calc_norm_conf_each_sample(self, distributions, samples):
        features = samples.T  # (n_features, n_samples)
        log_likelihoods = sum(
            gm.score_feature(features[i])
            for i, gm in enumerate(distributions)
        )
        log_likelihoods_max = sum(
            float(gm.score_feature(np.array([gm.mean])))
            for gm in distributions
        )
        return np.exp(log_likelihoods - log_likelihoods_max)
