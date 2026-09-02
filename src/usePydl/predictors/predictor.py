import numpy as np
from pydl85 import DL85Predictor

import CONFIG
from src.usePydl.error_fun import IntervalSizesError, ClusterCoherenceError
from src.usePydl.leaf import ReturnIDSandPROB
from src.samplers.single_gaussian import SingleGaussian1DSampler
from usePydl.predictors.helpers.interval import Interval, add_constraint_union

from usePydl.predictors.helpers.tree import Tree, remap_tree

COMBINE_FEAT = False


class Predictor:
    n_samples = None

    def __init__(self, feat_hist, weights, max_depth, min_sup, time, use_native_error=True):
        if weights is None:
            weights = CONFIG.GLOBAL_CHUNK_INFO.feature_importance
        print('starting dl predictor...')
        error_fun = ClusterCoherenceError(feat_hist.samples, weights)
        leaf_val = ReturnIDSandPROB(feat_hist.get_first_hist_depth_0().samples.shape[0])
        self.dl_predictor = None
        self.dl_predictor = DL85Predictor(error_function=error_fun,
                                          leaf_value_function=leaf_val,
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)

        self.dl_predictor.fit(feat_hist.get_splits_array().T)
        self.error = self.dl_predictor.error_
        tree = Tree(tree=self.dl_predictor.tree_)
        tree.tree = remap_tree(tree.tree, feature_history=feat_hist)
        feat_hist.tree = tree
        self.tree = tree
        self.feature_history = feat_hist


    def predict(self, samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def get_tree_dict(self):
        complete = self.feature_history.get_complete_tree()
        return complete.tree if complete is not None else self.tree.tree

    def gen_data_from_single_tree(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        complete_tree = self.feature_history.get_complete_tree()
        paths = complete_tree.get_all_paths()
        intervals_each_path = get_intervals_from_paths(paths)
        n_feats = len(CONFIG.GLOBAL_CHUNK_INFO.feature_importance)

        #path has 'directions', 'leaf_vals' and 'splits'
        if len(paths) == 0:
            raise ValueError("Tree has no paths")


        probs_each_path = np.array([path['leaf_vals']['rel_prob'] for path in paths])
        prob_sum = np.sum(probs_each_path)
        if prob_sum > 0:
            probs_each_path /= prob_sum
        else:
            probs_each_path = np.ones(len(probs_each_path)) / len(probs_each_path)

        all_new_samples = np.array([])
        indx_list_n = np.random.choice(len(paths), n, p=probs_each_path)
        indx, counts = np.unique(indx_list_n, return_counts=True)
        sample_count = 0

        for idx, count in zip(indx, counts):
            gen_feat_matrix = np.zeros((n_feats, count))
            intervals_each_feature = intervals_each_path[idx]
            sample_count += count
            for feat in range(n_feats):
                intervals = intervals_each_feature[feat]
                gen_feat_matrix[feat] = SingleGaussian1DSampler.sample_from_interval(intervals, count)
            if all_new_samples.size > 0:
                all_new_samples = np.vstack((all_new_samples, gen_feat_matrix.T))
            else:
                all_new_samples = gen_feat_matrix.T

        print(f"Generated {sample_count} samples.")
        if all_new_samples.size == 0:
            return all_new_samples
        return all_new_samples


def get_intervals_from_paths(paths):
    chunk_info = CONFIG.GLOBAL_CHUNK_INFO
    paths_intervals_feats = []
    for path in paths:

        splits = path['splits']
        directions = path['directions']
        intervals = []
        for j, d in enumerate(directions):
            split_obj = splits[j]
            if d == 'L':
                intervals.append(split_obj.left_interval())
            else:
                intervals.append(split_obj.right_interval())
        feat_inter = {}
        paths_intervals_feats.append(feat_inter)
        for feat in range(len(chunk_info.feature_importance)):
            combined_inter = [Interval(chunk_info.processed_feat_min[feat], chunk_info.processed_feat_max[feat], 'closed')]
            for inter in intervals:
                inter = inter.get(feat, None)
                if inter is not None:
                    combined_inter = add_constraint_union(combined_inter, inter)
            feat_inter[feat] = combined_inter
    return paths_intervals_feats

