import numpy as np
from pydl85 import DL85Predictor
from src.usePydl.leaf import get_leaf_vals

class Predictor:
    def __init__(self,samples_bin,error_fun,leaf_val,max_depth,min_sup,time):
        self.samples_bin = samples_bin
        self.predictor = DL85Predictor(error_function=error_fun,leaf_value_function=leaf_val,
                                       max_depth=max_depth,min_sup=min_sup, time_limit=time)
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

    def _generate_new_leaf_samples(self, n,distributions,conf_tresh):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def get_distributions(self, features):
        distr_funs = []
        for feat in features:
            distr_funs.append(self.get_distr(feat))
        return distr_funs  # [dstr-f1, dstr-f2, dstr-f3,...]

    def generate_new_data(self, n_new_samples: int = 100, conf_tresh: float = 0.8,keep_normal_distr: bool = False) -> np.ndarray:
        leafs = get_leaf_vals(self.predictor.tree_)
        samples_counts_leaf = []
        distrbution_leafs = []
        for leaf in leafs:
            distrbution_leafs.append(leaf['value']['distr'])
            samples_counts_leaf.append(leaf['value']['count'])
        new_samples = []
        total_leaf_s_count = np.sum(samples_counts_leaf)
        for i, count in enumerate(samples_counts_leaf):
            if keep_normal_distr:
                n = int((count/total_leaf_s_count) * n_new_samples)
            else:
                n = int(n_new_samples/total_leaf_s_count)
            samples = self._generate_new_leaf_samples(n,distrbution_leafs[i],conf_tresh)
            for sample in samples:
                new_samples.append(sample)
        return np.array(new_samples)

    def get_error_sample(self, distributions, samples):
        error = 1 - self.calc_norm_conf_each_sample(distributions, samples)
        return error




