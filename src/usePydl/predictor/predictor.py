import numpy as np
from pydl85 import DL85Predictor
from src.usePydl.leaf import get_leaf_vals
from src.usePydl.error_fun import predictor_error, reduce_interval_sizes
from src.samplers.load_samplers import get_sampler_class
from src.usePydl.leaf import default_leaf_val

class Predictor:
    def __init__(self, boolean_splits,samples,sampler_types, max_depth, min_sup, time):
        self.samples_bin = boolean_splits
        self.dl_predictor = DL85Predictor(error_function=reduce_interval_sizes(samples),
                                          leaf_value_function=default_leaf_val(samples,sampler_types),
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)
        self.fit()

    def fit(self):
        print("fitting_predictor")
        self.dl_predictor.fit(self.samples_bin)

    def load_new_data(self,samples_bin):
        self.samples_bin = samples_bin
        self.fit()

    def predict(self,samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def generate_new_data(self, n_new_samples: int = 100, conf_tresh: float = 0.8,mode: str = "keep_counts") -> np.ndarray:
        #the leafs at the moment store a lot of information which is all used here.
        #maybe simplify this by using the feature.active_splits dicts and jsut the dl_tree structure itself (since it tells us on which feature is split on)
        #build the intervals, create a sampler with it and sample
        leafs = get_leaf_vals(self.dl_predictor.tree_)
        samplers_x_leafs = [leaf["value"]["samplers_dict"] for leaf in leafs]
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

    def _generate_new_leaf_samples(self, n, samplers_dict, conf_thresh):
        from src.samplers.load_samplers import get_sampler_class

        if not samplers_dict or n <= 0:
            return []

        total_features = sum(len(group["indices"]) for group in samplers_dict.values())
        combined_features = np.zeros((n, total_features))

        for stype, group in samplers_dict.items():
            cls = get_sampler_class(stype)
            indices = group["indices"]
            samplers = group["samplers"]
            gen_feats = cls.generate_new_samples_for_all_features_of_this_type(n, conf_thresh, samplers)

            for i, idx in enumerate(indices):
                combined_features[:, idx] = gen_feats[:, i]

        return combined_features




