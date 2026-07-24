from typing import List, Optional
import numpy as np


DESCRETE_PERCENTILE = 5


def create_new_history(samples):
    feat_history = FeatureHistory(samples=samples)
    return feat_history

def extend_history(old_history: FeatureHistory,new_samples):
    new_history = FeatureHistory(samples=new_samples)
    old_history.add_future(new_history)
    new_history.set_past(old_history)
    return old_history



class FeatureHistory:
    def __init__(self, samples):
        self.feature_info_list = None
        self.past = None
        self.future_list = []
        self.create_current_history(samples)


    def create_current_history(self, samples):
        self.feature_info_list = [FeatureInfo(feat) for feat in samples.T] #will auto scale each feature

    def set_past(self,history: FeatureHistory):
        self.past = history

    def add_future(self,history: FeatureHistory):
        self.future_list.append(history)

    def get_sample_array_from_history(self):
        return np.array([l.feature_array for l in self.feature_info_list]).T


    def get_rescaled_sample_based_on_history(self, samples):
        features = samples.T
        rescaled_features = np.array([self.feature_info_list[i].reverse_scale(f) for i, f in enumerate(features)])
        return rescaled_features.T


class FeatureInfo:  # represents a single column of data
    def __init__(self, feature_data: np.ndarray):
        self.min_val = 0
        self.max_val = 0

        self.featureType = ''
        self.unique_values  = None

        self.feature_array = self.scale_and_standardize(feature_data)
        self.check_discrete()
        self.dependent_feat = []

    def scale_and_standardize(self, raw_feature_data):
        if raw_feature_data is None:    return None
        num_arr = make_num(raw_feature_data)
        self.min_val = np.min(num_arr)
        self.max_val = np.max(num_arr)
        if self.max_val - self.min_val == 0:    return np.zeros(len(num_arr))
        return np.array((num_arr - self.min_val) / (self.max_val - self.min_val))


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