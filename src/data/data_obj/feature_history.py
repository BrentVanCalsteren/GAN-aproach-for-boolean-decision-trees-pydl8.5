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
    new_history = FeatureHistory(samples=new_samples, is_scale=False)
    old_history.add_future(new_history)
    new_history.set_past(old_history)
    new_history.leaf_id = l_id
    new_history.depth = old_history.depth + 1
    return new_history



class FeatureHistory:
    def __init__(self, samples, is_scale=True):
        self.feature_info_list = None
        self.past = None
        self.future = []

        self.splits_obj: Splits = None
        self.samples = None

        self.depth = 0
        self.tree:Tree = None
        self.leaf_id = None

        self.create_current_history(samples, is_scale)


    def create_current_history(self, samples, is_scale):
        self.feature_info_list = [FeatureInfo(feat, is_scale) for feat in samples.T] #will auto scale each feature
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


    def get_rescaled_sample_based_on_history(self, samples):
        features = samples.T
        rescaled_features = np.array([self.feature_info_list[i].reverse_scale(f) for i, f in enumerate(features)])
        return rescaled_features.T

    def creat_splits(self, max_num_splits_each_feature=None):
        if max_num_splits_each_feature is None:
            max_num_splits_each_feature = 5
        self.splits_obj = Splits(max_splits_each_feature=max_num_splits_each_feature, samples=self.samples)
        for feature_info in self.feature_info_list:
            self.splits_obj.create_splits_from_feature(feature_info.feature_array,feature_info.featureType)

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
                        hist_tree = remap_tree(hist_tree,feature_history=hist)
                        tree = extend_tree(tree, hist_tree, hist.leaf_id)
                        new_futures += hist.future
                    else: print('no tree dict')
                else: print('no tree obj')
            futures = new_futures
        return Tree(tree=tree)


class FeatureInfo:  # represents a single column of data
    def __init__(self, feature_data: np.ndarray, is_scale=True):
        self.min_val = 0
        self.max_val = 0

        self.featureType = ''
        self.unique_values  = None
        self.feature_array = self.scale_and_standardize(feature_data, is_scale)
        self.check_discrete()
        self.dependent_feat = []

    def scale_and_standardize(self, raw_feature_data, is_scale):
        if raw_feature_data is None:    return None
        num_arr = make_num(raw_feature_data)
        if is_scale:
            self.min_val = np.min(num_arr)
            self.max_val = np.max(num_arr)
            if self.max_val - self.min_val == 0:    return np.zeros(len(num_arr))
            return np.array((num_arr - self.min_val) / (self.max_val - self.min_val))
        else:
            return np.array(num_arr)


    def reverse_scale(self, scaled_arr: np.ndarray):
        if self.max_val - self.min_val == 0:
            return np.full_like(scaled_arr, self.min_val)
        origin = scaled_arr * (self.max_val - self.min_val) + self.min_val
        if self.featureType == 'discrete':
            sorted_B = np.sort(self.unique_values)
            idx = np.searchsorted(sorted_B,origin)
            idx = np.clip(idx, 1, len(sorted_B) - 1)
            left = sorted_B[idx - 1]
            right = sorted_B[idx]
            use_left = np.abs(origin - left) < np.abs(origin - right)
            mapped = np.where(use_left, left, right)
            return mapped
        else: return origin

    def check_discrete(self):
        unique_vals = np.unique(self.feature_array)
        if len(unique_vals) <= (self.feature_array.shape[0] * DESCRETE_PERCENTILE // 100):
            self.featureType = 'discrete'
            self.unique_values = unique_vals
        else:
            self.featureType = 'continuous'

def make_num(raw_feature_data):
    try:
        num_arr = np.asarray(raw_feature_data, dtype=float)
    except (ValueError, TypeError):
        unique_values = np.unique(raw_feature_data)
        indexes = {val: idx for idx, val in enumerate(unique_values)}
        num_arr = np.array([indexes[val] for val in raw_feature_data])
        #maybe store the original strings? but would never need them (always want it to be numbers)
    return num_arr