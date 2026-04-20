import numpy as np
from typing import List
from src.usePydl.predictor.predictor_obj import Predictor
from src.usePydl.error_fun import mse_error
from src.usePydl.leaf import leaf_val, get_leaf_vals
from src.distributions.uniform import Uniform_distr

class UNiPredictor(Predictor):
    #Fit Complexity: O(n·d)
    #score complexity: 	O(d)
    def __init__(self,samples: np.ndarray,samples_bin: np.ndarray,max_depth: int = 3,min_sup: int = 2,time: int = 100):
        self.samples = samples
        super().__init__(
            samples_bin=samples_bin,
            error_fun=mse_error(samples),
            leaf_val=leaf_val(self, samples),
            max_depth=max_depth,
            min_sup=min_sup,
            time=time
        )
        self.generate_tree()

    def get_distr(self, feature_array):
        uni = Uniform_distr()
        uni.fit(feature_array.reshape(-1, 1))
        return uni

    def generate_new_data(self, n_new_samples: int = 100, conf_tresh: float = 0.8) -> np.ndarray:
        leafs = get_leaf_vals(self.predictor.tree_)
        samples_counts_leaf = []
        distrbution_leafs = []
        for leaf in leafs:
            distrbution_leafs.append(leaf['value']['distr'])
            samples_counts_leaf.append(leaf['value']['count'])
        new_samples = []
        total_leaf_s_count = np.sum(samples_counts_leaf)
        for i, count in enumerate(samples_counts_leaf):
            n = int((count/total_leaf_s_count) * n_new_samples)
            samples = _generate_new_leafsamples(n,distrbution_leafs[i],conf_tresh)
            for sample in samples:
                new_samples.append(sample)
        return np.array(new_samples)

def _generate_new_leafsamples(n,distributions: List[Uniform_distr],conf_tresh):
    samples_above_tresh = []
    while len(samples_above_tresh) < n:
        features = []
        for distr in distributions:
            gen = distr.sample(n_samples=100) #returns ([[10*[f1]],[10*[f2]],...] , Label)
            feat = []
            for point in gen:
                feat.append(point[0])
            features.append(feat)
        features = np.array(features)
        good_features = []
        feature_x_sample_prob_matrix = calc_norm_conf_feature_x_sample(distributions, features)
        for i,feature in enumerate(feature_x_sample_prob_matrix):
            indices = np.argsort(feature)
            good_points = features[i,indices]
            good_features.append(good_points)
        samples = np.array(good_features).T
        sample_prob = calc_norm_conf_each_sample(distributions, samples)
        for i,prob in enumerate(sample_prob):
            if prob >= conf_tresh:
                samples_above_tresh.append(samples[i])
    return np.array(samples_above_tresh)[:n]



def calc_norm_conf_each_sample(distributions, samples):
    lows = np.array([d.min[0] for d in distributions])
    highs = np.array([d.max[0] for d in distributions])

    sample_x_feature = ((samples >= lows) & (samples <= highs)).astype(float)
    samples_ll = np.zeros(samples.shape[0])
    for i, sample in enumerate(sample_x_feature):
        samples_ll[i] = np.sum(sample)
    ll_max = 1 * len(distributions)

    # Normalise
    confidence = samples_ll/ll_max
    return confidence


def calc_norm_conf_feature_x_sample(distributions: List[Uniform_distr], features: np.ndarray) -> np.ndarray:
    lows = np.array([d.min[0] for d in distributions])
    highs = np.array([d.max[0] for d in distributions])

    #(n_features, n_samples)
    inside = (features >= lows[:, np.newaxis]) & (features <= highs[:, np.newaxis])
    return inside.astype(float)

