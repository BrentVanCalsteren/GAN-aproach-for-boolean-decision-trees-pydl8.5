import copy
from typing import List, Optional
import numpy as np

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
    def __init__(self, samples, chunkInfo=None):
        self.feature_info_list = None
        self.chunkInfo = chunkInfo
        self.past = None
        self.future = []

        self.splits_obj: Splits = None
        self.samples = None

        self.depth = 0
        self.tree:Tree = None
        self.leaf_id = None
        self.pred_error = np.inf

        self.create_current_history(samples)


    def create_current_history(self, samples):
        self.feature_info_list = [FeatureInfo(feat,feat_id, self.chunkInfo) for feat_id, feat in enumerate(samples.T)] #will auto scale each feature
        self.samples = np.array([l.feature_array for l in self.feature_info_list]).T

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


    def creat_splits(self, total_num_splits=50, weight_of_each_feature=None):
        n_feats = len(self.feature_info_list)
        feat_splits_num = []
        if weight_of_each_feature is None:
            weight_of_each_feature = self.get_feature_weights()

        for i in range(n_feats):
            w = weight_of_each_feature[i] if i < len(weight_of_each_feature) else (1.0 / float(n_feats))
            a = max(1, int(total_num_splits * w))
            feat_splits_num.append(a)

        self.splits_obj = Splits(max_splits_each_feature=feat_splits_num, samples=self.samples)
        for feature_info in self.feature_info_list:
            self.splits_obj.create_splits_from_feature(feature_info.feature_array, feature_info.feat_id)

    def get_feature_weights(self, mode: str = 'uniform', focus_on_percentage: float = 0.5) -> np.ndarray:
        n_feats = len(self.feature_info_list)
        if n_feats == 0:
            return np.array([])
        if mode == 'random':
            weights = np.random.uniform(0.1, 1.0, size=n_feats)
        else:
            if self.chunkInfo is not None and self.chunkInfo.feature_importance is not None:
                raw_imp = np.asarray(self.chunkInfo.feature_importance).flatten()
                if len(raw_imp) >= n_feats:
                    weights = raw_imp[:n_feats].copy()
                else:
                    weights = np.ones(n_feats, dtype=float) / float(n_feats)
                    weights[:len(raw_imp)] = raw_imp
            else:
                weights = np.ones(n_feats, dtype=float)
        weights = np.maximum(0.0, np.asarray(weights, dtype=float))
        sum_w = np.sum(weights)
        if sum_w > 0:
            weights /= sum_w
        else:
            weights = np.ones(n_feats, dtype=float) / float(n_feats)

        if 0.0 < focus_on_percentage < 1.0:
            k = max(1, int(np.ceil(n_feats * focus_on_percentage)))
            top_k_indices = np.argsort(weights)[-k:]
            mask = np.zeros(n_feats, dtype=bool)
            mask[top_k_indices] = True
            weights[~mask] = 0.0
            sum_top = np.sum(weights)
            if sum_top > 0:
                weights /= sum_top
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


class FeatureInfo:  # represents a single column of data
    def __init__(self, feature_data: np.ndarray,feat_id, chunkInfo):
        self.feat_id = feat_id
        self.chunkInfo = chunkInfo
        self.feature_array = feature_data
        self.dependent_feat = []

