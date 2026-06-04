import numpy as np
import re
from src.data.split_generator import get_best_splits

DESCRETE_PERCENTILE = 5


class Feature:  # represents a single column of data
    def __init__(self, raw_feature_data: np.ndarray, sample_class):
        self.min_val = 0
        self.max_val = 0
        self.sample_class = sample_class
        self.active_splits = {}
        self.errors = []
        self.isDiscrete = False
        self.feature_info = [None, None]  # tells predictors which sampler to use, how many bool splits

        self.feature_array = self.scale_and_standardize(raw_feature_data)
        self.check_discrete()
        self.dependent_feat = []

    def scale_and_standardize(self, raw_feature_data):
        if raw_feature_data is None:
            return None
        try:
            num_arr = np.asarray(raw_feature_data, dtype=float)
        except (ValueError, TypeError):
            # Convert chars to index
            unique_values = np.unique(raw_feature_data)
            indexes = {val: idx for idx, val in enumerate(unique_values)}
            num_arr = np.array([indexes[val] for val in raw_feature_data])

        self.min_val = num_arr.min()
        self.max_val = num_arr.max()

        if self.max_val - self.min_val == 0:
            return np.zeros(len(num_arr))
        return np.array((num_arr - self.min_val) / (self.max_val - self.min_val))

    def reverse_scale(self, scaled_arr: np.ndarray):
        if self.max_val - self.min_val == 0:
            return np.full_like(scaled_arr, self.min_val)
        return scaled_arr * (self.max_val - self.min_val) + self.min_val

    def check_discrete(self):
        unique_vals = np.unique(self.feature_array)
        if len(unique_vals) <= (self.feature_array.shape[0] * DESCRETE_PERCENTILE // 100):
            self.isDiscrete = True
            print('discrete feature')
            self.feature_info[0] = unique_vals
            return

    def feature_to_samples(self):
        return self.feature_array.reshape(-1, 1)

    def get_splits_bool(self):
        return np.array(list(self.active_splits.values())).squeeze()

    def get_splits_val(self):
        return np.array(list(self.active_splits.keys())).squeeze()

    def map_feature_to_active_splits(self, new_feature):
        keys = list(self.active_splits.keys())
        splits = []
        for key in keys:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", key)
            vals = [float(x) for x in matches]
            if 'bigger' in key:
                result = np.where(new_feature > vals[0], 1, 0)
            elif 'even' in key:
                result = np.where(new_feature == vals[0], 1, 0)
            elif 'smaller' in key:
                result = np.where(new_feature < vals[0], 1, 0)
            elif 'interval' in key:
                result = np.where((new_feature >= vals[0]) & (new_feature <= vals[1]), 1, 0)
            else:
                print('invalid key')
                return None
            splits.append(result)
        return np.array(splits)

    def create_all_independent_splits(self, n_splits):
        # Delegate to fast, memory-efficient generator
        self.active_splits = get_best_splits(self.feature_array, n_splits, self.isDiscrete)
        self.feature_info[1] = len(self.active_splits)
