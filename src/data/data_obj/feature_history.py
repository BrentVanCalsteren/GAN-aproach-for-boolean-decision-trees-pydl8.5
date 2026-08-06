import copy
from typing import List, Any
import numpy as np

import CONFIG
from data.data_obj.splits import Splits
from usePydl.predictors.tree import Tree, remap_tree, extend_tree

DESCRETE_PERCENTILE = 5


def create_new_history(samples):
    feat_history = FeatureHistory(samples=samples)
    return feat_history

def extend_history(new_samples, old_history: FeatureHistory, l_id):
    new_history = FeatureHistory(samples=new_samples, chunkInfo=old_history.chunkInfo)
    old_history.add_future(new_history)
    new_history.set_past(old_history)
    new_history.leaf_id = l_id
    new_history.depth = old_history.depth + 1
    return new_history



class FeatureHistory:
    def __init__(self, samples, chunkInfo=None, stays_in_memory=False):
        self.feature_info_list = None
        self.chunkInfo = chunkInfo
        self.past = None
        self.future = []

        self.stays_in_memory = stays_in_memory

        self.splits_obj: Splits = None
        self.samples = None
        self.feat_min_vals = None
        self.feat_max_vals = None
        self.corr_matrix = None

        self.depth = 0
        self.tree:Tree = None
        self.leaf_id = None
        self.pred_error = np.inf

        self.create_current_history(samples)


    def create_current_history(self, samples):
        self.feature_info_list = [0] * samples.shape[1]
        self.samples = samples
        if self.samples is not None and self.samples.size > 0:
            self.feat_min_vals = np.min(self.samples, axis=0)
            self.feat_max_vals = np.max(self.samples, axis=0)
            self.corr_matrix = np.corrcoef(self.samples, rowvar=False)
            np.nan_to_num(self.corr_matrix, copy=False, nan=0.0)
        else:
            self.feat_min_vals = np.array([])
            self.feat_max_vals = np.array([])

    def reduce_memory(self):
        if self.stays_in_memory:
            return
        self.samples = None
        self.splits_obj = None
        if self.future:
            for child in self.future:
                child.reduce_memory()
        self.past = None
        self.future = []

    def set_past(self,history: FeatureHistory):
        self.past = history

    def add_future(self,history: FeatureHistory):
        self.future.append(history)

    def get_first_hist_depth_0(self):
        hist = self
        depth = hist.depth
        while depth > 0:
            hist = hist.past
            depth = hist.depth
        return hist

    def get_sample_array_from_history(self):
        return self.samples


    def creat_splits(self, weight_of_each_feature=None):
        n_feats = len(self.feature_info_list)
        total_num_splits = n_feats*CONFIG.AVG_BOOL_SPLITS_EACH_FEATURE
        total_num_splits = CONFIG.MAX_SPLITS
        feat_splits_num = []
        if weight_of_each_feature is None:
            weight_of_each_feature = self.get_feature_weights()

        for i in range(n_feats):
            w = weight_of_each_feature[i]
            a = max(1, int(total_num_splits * w))
            feat_splits_num.append(a)

        self.splits_obj = Splits(max_splits_each_feature=feat_splits_num, samples=self.samples, weights=self.chunkInfo.feature_importance)
        for i, _ in enumerate(self.feature_info_list):
            self.splits_obj.create_splits_from_feature(self.samples.T[i,:], i)

    def get_feature_weights(self, mode: str = 'uniform', focus_on=None):
        n_feats = len(self.feature_info_list)
        if n_feats == 0: return np.array([])

        if mode == 'random':
            weights = np.random.uniform(0.1, 1.0, size=n_feats)
        else:
            if self.chunkInfo.feature_importance is not None:
                weights = np.asarray(self.chunkInfo.feature_importance).flatten()
            else:   weights = np.ones(n_feats, dtype=float)

        sum_w = np.sum(weights)
        if sum_w > 0: weights /= sum_w
        else:  weights = np.ones(n_feats, dtype=float) / float(n_feats)

        if focus_on is not None:
            mask = np.zeros(n_feats, dtype=bool)
            if isinstance(focus_on, float) and 0.0 <= focus_on <= 1.0:
                k = max(1, int(np.ceil(n_feats * float(focus_on))))
                top_k_indices = np.argsort(weights)[-k:]
                mask[top_k_indices] = True
            else:
                valid_ids = [int(f) for f in focus_on if 0 <= int(f) < n_feats]
                if len(valid_ids) > 0: mask[valid_ids] = True
                else: mask[:] = True

            weights[~mask] = 0.0
            sum_top = np.sum(weights)
            if sum_top > 0: weights /= sum_top
            elif np.any(mask): weights[mask] = 1.0 / float(np.sum(mask))
        return weights

    def get_splits(self):
        return self.splits_obj.get_splits()

    def get_feat_split_result(self, bool_feat_id: int):
        real_feat = self.splits_obj.feature_index_array[bool_feat_id]
        value = self.splits_obj.values[bool_feat_id]
        return real_feat, value

    def get_local_tree(self):
        return self.tree

    def get_complete_tree(self):
        root = self
        futures = root.future
        tree = copy.deepcopy(self.tree.tree)
        if tree is None: return None
        tree = remap_tree(tree, feature_history=self)
        while len(futures) > 0:
            new_futures = []
            for hist in futures:
                if hist.tree is not None:
                    if hist.tree.tree is not None:
                        hist_tree = copy.deepcopy(hist.tree.tree)
                        tree = extend_tree(tree, hist_tree, hist.leaf_id)
                        new_futures += hist.future
                    else: print('no tree dict')
                else: print('no tree obj')
            futures = new_futures
        return Tree(tree=tree)

