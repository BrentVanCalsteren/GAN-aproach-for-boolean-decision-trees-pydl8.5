import numpy as np
from pydl85 import DL85Predictor
from src.usePydl.leaf import get_leaf_vals

class Predictor:
    def __init__(self,samples_bin,error_fun,leaf_val,max_depth,min_sup,time):
        self.samples_bin = samples_bin
        self.predictor = DL85Predictor(error_function=error_fun,
                                       leaf_value_function=leaf_val,
                                       max_depth=max_depth,
                                       min_sup=min_sup,
                                       time_limit=time,
                                       max_error=np.inf)
    def generate_tree(self):
        self.predictor.fit(self.samples_bin)

    def load_new_data(self,samples_bin):
        self.samples_bin = samples_bin
        self.generate_tree()

    def predict(self,samples_bin):
        return self.predictor.predict(samples_bin)

    def calc_norm_conf_each_sample(self, distributions, samples):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def get_distr(self, feature_array):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def get_distributions(self, features):
        distr_funs = []
        for feat in features:
            distr_funs.append(self.get_distr(feat))
        return distr_funs  # [dstr-f1, dstr-f2, dstr-f3,...]

    def generate_new_data(self, n_new_samples: int = 100, conf_tresh: float = 0.8,mode: str = "keep_counts") -> np.ndarray:
        leafs = get_leaf_vals(self.predictor.tree_)
        distributions_x_leafs = [leaf["value"]["distr"] for leaf in leafs]
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
            if samples_in_leaf[i] > 3:
                 new_samples.extend(self._generate_new_leaf_samples(ns[i],distributions_x_leafs[i],conf_tresh))
        return np.array(new_samples)

    def _generate_new_leaf_samples(self, n, distributions, conf_thresh):
        samples_above_thresh = []
        while len(samples_above_thresh) < n:
            gen_feats = np.array([distr.sorted_samples(n=100) for distr in distributions])  # (n_features, 100)
            samples = gen_feats.T  #(100, n_features)
            confidence = self.calc_norm_conf_each_sample(distributions, samples)
            samples_above_thresh.extend(samples[confidence >= conf_thresh])
        return np.array(samples_above_thresh)[:n]

    def get_error_sample(self, distributions, samples):
        error = 1 - self.calc_norm_conf_each_sample(distributions, samples)
        return error




