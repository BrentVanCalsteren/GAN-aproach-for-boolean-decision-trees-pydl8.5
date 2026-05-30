from typing import List
from src.usePydl.predictor.predictor import Predictor
from src.usePydl.error_fun import predictor_error
from src.usePydl.leaf import default_leaf_val
from sklearn.mixture import GaussianMixture
import numpy as np

class GaussianMultiPredictor(Predictor):
    #Fit Complexity: O(k·n·d²·EM algorithm used)k=gaus_components,n=number samples,d=num features
    #score complexity: 	O(k·d)
    def __init__(self, samples, boolean_splits, max_depth=3, min_sup=1, time=100):
        self.samples = samples
        super().__init__(
            boolean_splits=boolean_splits,
            error_fun=predictor_error(self, samples),
            leaf_val=default_leaf_val(self,samples),
            max_depth=max_depth,
            min_sup=min_sup,
            time=time
        )
        self.fit()

    def get_distr(self, feature_array, n_components=1):
        gm = GaussianMixture(n_components=n_components, covariance_type='full')
        gm.fit(feature_array.reshape(-1, 1))
        return gm
#TODO update

    def calc_norm_conf_each_sample(self, distributions: List[GaussianMixture], samples: np.ndarray) -> np.ndarray:
        features = samples.T  # (n_features, n_samples)
        log_likelihoods = np.zeros(samples.shape[0])
        log_likelihoods_max = 0.0
        for i, gm in enumerate(distributions):
            log_likelihoods += gm.score_samples(features[i].reshape(-1, 1))

            # peak likelihood: mean of the most weighted component
            dominant_mean = gm.means_[np.argmax(gm.weights_)]
            log_likelihoods_max += gm.score_samples(dominant_mean.reshape(1, -1))[0]

        return np.exp(log_likelihoods - log_likelihoods_max)



