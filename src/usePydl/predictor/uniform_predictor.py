from typing import List

import numpy as np

from src.usePydl.predictor.predictor_obj import Predictor
from src.usePydl.error_fun import predictor_error,mse_error,mae_error
from src.usePydl.leaf import default_leaf_val
from src.distributions.uniform import Uniform_distr

class UNiPredictor(Predictor):
    #Fit Complexity: O(n·d)
    #score complexity: 	O(d)
    def __init__(self,samples: np.ndarray,samples_bin: np.ndarray,max_depth: int = 3,min_sup: int = 1,time: int = 100):
        self.samples = samples
        super().__init__(
            samples_bin=samples_bin,
            error_fun=predictor_error(self, samples),
            leaf_val=default_leaf_val(self, samples),
            max_depth=max_depth,
            min_sup=min_sup,
            time=time
        )
        self.generate_tree()

    def get_distr(self, feature_array):
        uni = Uniform_distr()
        uni.fit(feature_array.reshape(-1, 1))
        return uni

    #used in default error function
    def calc_norm_conf_each_sample(self, distributions: List[Uniform_distr], samples: np.ndarray):
        features = samples.T
        scores = np.array([dist.score_feature(features[i]) for i, dist in enumerate(distributions)]).T #shape scores (n_samples,n_features)
        return np.mean(scores, axis=1)



