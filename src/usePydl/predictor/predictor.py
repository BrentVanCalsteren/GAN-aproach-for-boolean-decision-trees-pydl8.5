import numpy as np
from pydl85 import DL85Predictor
from src.usePydl.leaf import get_leaf_vals
from src.usePydl.error_fun import predictor_error
from src.usePydl.leaf import default_leaf_val

class Predictor:
    def __init__(self, boolean_splits,samples,sampler_types, max_depth, min_sup, time):
        self.samples_bin = boolean_splits
        self.dl_predictor = DL85Predictor(error_function=predictor_error(samples,sampler_types),
                                          leaf_value_function=default_leaf_val(samples,sampler_types),
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)
        self.fit()

    def fit(self):
        self.dl_predictor.fit(self.samples_bin)

    def load_new_data(self,samples_bin):
        self.samples_bin = samples_bin
        self.fit()

    def predict(self,samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def generate_new_data(self, n_new_samples: int = 100, conf_tresh: float = 0.8,mode: str = "keep_counts") -> np.ndarray:
        leafs = get_leaf_vals(self.dl_predictor.tree_)
        samplers_x_leafs = [leaf["value"]["samplers"] for leaf in leafs]
        samples_in_leaf = np.array([leaf["value"]["count"] for leaf in leafs])
        total_count = samples_in_leaf.sum()
        new_samples = []
        ns = None
        if mode == "keep_counts":
            ns = ((samples_in_leaf / total_count) * n_new_samples).astype(int)
        elif mode == "even":
            ns = np.full(len(leafs), n_new_samples // len(leafs))
        else:
            raise ValueError(f"Unknown mode")
        for i in range(len(leafs)):
            #if samples_in_leaf[i] > 3:
                 new_samples.extend(self._generate_new_leaf_samples(ns[i],samplers_x_leafs[i],conf_tresh))
        return np.array(new_samples)

    def _generate_new_leaf_samples(self, n, samplers, conf_thresh):
        samples_above_thresh = []
        while len(samples_above_thresh) < n:
            gen_feats = []
            score_matrix = []
            for sampler in samplers:
                gen_feat = sampler.sorted_samples(n=100)
                gen_feats.append(gen_feat)
                sampler.score_feat(gen_feat)
                score_matrix.append(sampler.score_feat(gen_feat))
            samples = np.array(gen_feats).T
            score_matrix = np.array(score_matrix).T
            good_scores = np.where(score_matrix >= conf_thresh)

        arr = np.array(samples_above_thresh)[:n]
        #clipped = np.clip(arr, 0, 1)
        return arr

    def get_error_sample(self, distributions, samples):
        error = 1 - self.calc_norm_conf_each_sample(distributions, samples)
        return error




