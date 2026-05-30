from typing import Dict

import numpy as np
import re

from usePydl.predictor.uniform_predictor import UNiPredictor
from dataLoader.divisive_clustering_1D import DivisiveCluster

DESCRETE_PERCENTILE = 5
class Feature:
    def __init__(self, raw_feature_data:np.ndarray):
        self.active_splits = {}
        self.isDiscrete = False
        self.feature_array = standardize_feature(raw_feature_data)
        self.check_descrete()

    def check_descrete(self):
        unique_vals = np.unique(self.feature_array)
        if len(unique_vals) <= (self.feature_array.shape[0] * DESCRETE_PERCENTILE // 100):
            self.isDiscrete = True

    def get_single_value_splits(self, n_splits, descrete_percentile = 5) -> Dict[str, np.ndarray]:
        def generate_all_possible_splits():
            splits = {}
            unique_vals = np.unique(self.feature_array)
            if self.isDiscrete:
                def split_discrete():
                    mapped = {}
                    for val in unique_vals:
                        result = np.where(self.feature_array == val, 1, 0)
                        mapped[f'even_{val}'] = result
                    return mapped

                splits = split_discrete()
            else:
                def splits_continue():
                    mapped = {}
                    for val in unique_vals:
                        result = np.where(self.feature_array <= val, 0, 1)
                        mapped[f'bigger_{val}'] = result
                    return mapped

                splits = splits_continue()
            return splits

        all_splits = generate_all_possible_splits()
        if self.isDiscrete:
            splits = self.remove_duplicate_vals(all_splits)
            keys = list(splits.keys())
            vals = list(splits.values())
            for i in range(len(keys)):
                self.active_splits[keys[i]] = vals[i]
            return (splits, [0] * len(splits))
        return self.calc_best_splits_with_entropy_score(all_splits, n_splits)

    def get_cluster_interval_splits(self, n_splits, check_cluster_configs=20) -> Dict[str, np.ndarray]:
        if self.isDiscrete:
            return ({},[])
        all_splits = {}
        cluster = DivisiveCluster()
        for i in range(2,check_cluster_configs):
            cluster.max_depth = i
            cluster.fit(self.feature_array)
            intervals = cluster.get_clusters()
            for interval in intervals:
                result = np.where((self.feature_array >= interval[0]) & (self.feature_array <= interval[1]), 1, 0)
                all_splits[f'interval_{interval[0]}_{interval[1]}'] = result
        return self.calc_best_splits_with_entropy_score(all_splits, n_splits)

    def get_percentile_binning(self, n_splits, check_different_percentages=20) -> Dict[str, np.ndarray]:
        if self.isDiscrete:
            return ({},[])
        all_splits = {}
        for i in range(2, check_different_percentages):
            start = 100//i
            percentages = []
            j = 0
            while start < 100:
                percentages.append(start)
                start = start+start
                j+=1
            percentages.append(100)
            thresholds = np.percentile(self.feature_array, percentages)
            for index, _ in enumerate(thresholds):
                if index < len(thresholds)-1:
                    result = np.where((self.feature_array >= thresholds[index]) & (self.feature_array <= thresholds[index+1]), 1, 0)
                    all_splits[f'interval_{thresholds[index]}_{thresholds[index+1]}'] = result

        return self.calc_best_splits_with_entropy_score(all_splits, n_splits)

    def remove_duplicate_vals(self, new_pos_splits : Dict[str,list]):
        splits_to_keep = {}
        new_vals = list(new_pos_splits.values())
        new_keys = list(new_pos_splits.keys())
        already_present = list(self.active_splits.values())
        for i in range(len(new_keys)):
            if check_if_sub_array(already_present,new_vals[i]):
                print("!Values is already present in the array!")
            else:
                splits_to_keep[new_keys[i]] = new_vals[i]
        return splits_to_keep

    def calc_best_splits_with_entropy_score(self, all_splits,n_splits) -> Dict[str, np.ndarray]:
        #uses pydl tree, generate a tree on all possible splits of depth 1,
        # the feature it choses will be the best split.
        all_splits = self.remove_duplicate_vals(all_splits)
        keys = np.array(list(all_splits.keys()))
        possible_splits = np.array(list(all_splits.values()))
        good_splits_dict = {}
        errors = []
        while len(good_splits_dict) < n_splits and possible_splits.shape[0] > 0:
            uni_pred = UNiPredictor(self.feature_to_samples(), possible_splits.T,max_depth=1,min_sup=1)
            tree = uni_pred.dl_predictor.tree_
            print(tree.keys())
            try:
                split_id = tree['feat']
                total_error = tree['left']['error'] + tree['right']['error']
            except KeyError:
                split_id = 0
                total_error = tree['error']
            print(f'found split: {keys[split_id]}')
            good_splits_dict[keys[split_id]] = possible_splits[split_id]
            errors.append(total_error)
            self.active_splits[keys[split_id]] = possible_splits[split_id]
            possible_splits = np.delete(possible_splits, split_id, axis=0)
            keys = np.delete(keys, split_id, axis=0)
        return (good_splits_dict, errors)

    def feature_to_samples(self):
        return self.feature_array.reshape(-1, 1)

    def map_to_splits(self, feature_to_map):
        keys = list(self.active_splits.keys())
        splits = []
        for key in keys:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", key)
            vals = [float(x) for x in matches]
            if 'bigger' in key:
                result = np.where(feature_to_map <= vals[0], 0, 1)
            elif 'even' in key:
                result = np.where(feature_to_map == vals[0], 1, 0)
            elif 'interval' in key:
                result = np.where((feature_to_map >= vals[0]) & (feature_to_map <= vals[1]), 1, 0)
            else:
                print('invalid key')
                return None
            splits.append(result)
        return np.array(splits)



##################"HELPERS

def standardize_feature(raw_feature_data):
    if raw_feature_data is None:
        return None
    try:
        num_arr = np.asarray(raw_feature_data, dtype=float)
        print("Converted numeric strings to float.")
        return scale(num_arr)
    except (ValueError, TypeError):
        num_arr = value_to_index(raw_feature_data)
        print("Converted chars to index")
        return scale(num_arr)

def scale(arr:np.ndarray):
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return np.zeros(len(arr))
    return np.array((arr - min_val) / (max_val - min_val))

def value_to_index(np_arr):
    unique_values = np.unique(np_arr)
    print(f"Unique vals: {unique_values}")
    indexes = {val: idx for idx, val in enumerate(unique_values)}  # Build mapping
    return np.array([indexes[val] for val in np_arr])

def check_if_sub_array(array,sub_array):
    for arr in array:
        if np.equal(arr, sub_array).all():
            return True
    return False

