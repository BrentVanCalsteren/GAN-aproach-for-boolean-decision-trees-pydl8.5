import numpy as np
from pydl85 import DL85Predictor
from src.usePydl.leaf import get_leaf_vals
from src.usePydl.error_fun import predictor_error, reduce_interval_sizes
from src.samplers.load_samplers import get_sampler_class
from src.usePydl.leaf import default_leaf_val
from src.samplers.load_samplers import get_sampler_class

class Predictor:
    def __init__(self,splits,samples, sampler_types, max_depth, min_sup, time):
        self.dl_predictor = DL85Predictor(error_function=reduce_interval_sizes(samples),
                                          leaf_value_function=default_leaf_val(samples,sampler_types),
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)
        self.dl_predictor.fit(splits)

    def predict(self,samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def generate_new_data(self, n_new_samples: int = 100, conf_tresh: float = 0.8,mode: str = "keep_counts") -> np.ndarray:
        leafs = get_leaf_vals(self.dl_predictor.tree_)
        samplers_x_leafs = [leaf["value"]["samplers_list"] for leaf in leafs]
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

    def _generate_new_leaf_samples(self, n, samplers_list, conf_thresh):
        total_features = samplers_list[0]["num_feat"]
        gen_feats = np.zeros((total_features, n))
        if n <= 0:
            return np.array([]).reshape((0,total_features))
        for info_dict in samplers_list:
            cls = info_dict["sample_class"]
            indices = info_dict["feat_ids"]
            samplers = info_dict["samplers"]
            cls.generate_new_samples_for_all_features_of_this_type(indices,gen_feats, conf_thresh, samplers)
        return gen_feats.T




