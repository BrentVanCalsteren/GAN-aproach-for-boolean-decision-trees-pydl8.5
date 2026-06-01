from typing import Dict

import numpy as np
import re
from usePydl.predictor.predictor import Predictor
from data.divisive_clustering_1D import DivisiveCluster

DESCRETE_PERCENTILE = 5
DIFFERENT_CLUSTERS_TO_CREATE = 10
DIFFERENT_PERCETILE_BINS_TO_CREATE = 10

class Feature: #is a d array of datapoints
    def __init__(self, raw_feature_data:np.ndarray):
        self.min_val = 0
        self.max_val = 0
        self.active_splits = {}
        self.errors = []
        self.isDiscrete = False
        self.feature_type = None #tells predictor which sampler to use
        feature_array = standardize_feature(raw_feature_data)
        self.feature_array = self.scale(feature_array)
        self.check_descrete()
        self.dependent_feat = []

    def scale(self, arr: np.ndarray):
        self.min_val = arr.min()
        self.max_val = arr.max()

        if self.max_val - self.min_val == 0:
            return np.zeros(len(arr))
        return np.array((arr - self.min_val) / (self.max_val - self.min_val))

    def reverse_scale(self, scaled_arr: np.ndarray):
        if self.max_val - self.min_val == 0:
            return np.full_like(scaled_arr, self.min_val)
        return scaled_arr * (self.max_val - self.min_val) + self.min_val


    def check_descrete(self):
        unique_vals = np.unique(self.feature_array)
        if len(unique_vals) <= (self.feature_array.shape[0] * DESCRETE_PERCENTILE // 100):
            self.isDiscrete = True
            self.feature_type = 'multinomial'
            return
        self.feature_type = 'multi_gaussian'

    def feature_to_samples(self):
        return self.feature_array.reshape(-1, 1)

    def get_splits_as_array(self):
        return np.array(list(self.active_splits.values())).squeeze()

    def map_feature_to_active_splits(self, new_feature):
        keys = list(self.active_splits.keys())
        splits = []
        for key in keys:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", key)
            vals = [float(x) for x in matches]
            if 'bigger' in key:
                result = np.where(new_feature <= vals[0], 0, 1)
            elif 'even' in key:
                result = np.where(new_feature == vals[0], 1, 0)
            elif 'interval' in key:
                result = np.where((new_feature >= vals[0]) & (new_feature <= vals[1]), 1, 0)
            else:
                print('invalid key')
                return None
            splits.append(result)
        return np.array(splits)

    ##############################################################################
    ############## creating boolean splits that are independent on other features#####################
    ##############################################################################
    def create_all_independent_splits(self, n_splits):
        current_indepent_types = ["1D_clustering to create intervals with",
                                  "take 1 value and split on it",
                                  "create intervals with percentile binning"]
        partly_splits = n_splits // len(current_indepent_types)
        self.indep_splits_single_value(partly_splits)
        self.indep_splits_cluster_interval(partly_splits)
        self.indep_splits_percentile_binning(partly_splits)

    def indep_splits_single_value(self, n_splits):
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
            splits = self.remove_duplicate_splits(all_splits)
            keys = list(splits.keys())
            vals = list(splits.values())
            for i in range(len(keys)):
                self.active_splits[keys[i]] = vals[i]
            return (splits, [0] * len(splits))
        self.calc_best_splits_with_entropy_score(all_splits, n_splits)

    def indep_splits_cluster_interval(self, n_splits):
        if self.isDiscrete: return
        all_splits = {}
        cluster = DivisiveCluster()
        for i in range(2, DIFFERENT_CLUSTERS_TO_CREATE):
            cluster.max_depth = i
            cluster.fit(self.feature_array)
            intervals = cluster.get_clusters()
            for interval in intervals:
                result = np.where((self.feature_array >= interval[0]) & (self.feature_array <= interval[1]), 1, 0)
                all_splits[f'interval_{interval[0]}_{interval[1]}'] = result
        self.calc_best_splits_with_entropy_score(all_splits, n_splits)

    def indep_splits_percentile_binning(self, n_splits):
        if self.isDiscrete: return
        all_splits = {}
        for i in range(2, DIFFERENT_PERCETILE_BINS_TO_CREATE):
            start = 100 // i
            percentages = []
            j = 0
            while start < 100:
                percentages.append(start)
                start = start + start
                j += 1
            percentages.append(100)
            thresholds = np.percentile(self.feature_array, percentages)
            for index, _ in enumerate(thresholds):
                if index < len(thresholds) - 1:
                    result = np.where(
                        (self.feature_array >= thresholds[index]) & (self.feature_array <= thresholds[index + 1]), 1, 0)
                    all_splits[f'interval_{thresholds[index]}_{thresholds[index + 1]}'] = result
        self.calc_best_splits_with_entropy_score(all_splits, n_splits)

    ##############################################################################
    ############## creating boolean splits that are dependent on other features#############
    # ############ maybe not neccecary since a pydl-tree itself experces feature dependency with
    # ########### what features it picks to split on (like it choses bool splits of feat 1, 3 of feat 2, ...
    # (need to check if this is correct assumption)? ##################################
    ############################################################################################"

    def create_all_dependent_splits(self,dependent_feats):
        #exploring dependency -> still need to implement
        current_depent_types = []

    ##############################################################################
    ############## function for picking best splits for top split #############
    ############################################################################################"

    def calc_best_splits_with_entropy_score(self, all_splits, n_splits):
        # uses pydl tree, generate a tree on all possible splits of depth 1, maybe this does not aply when feature
        # get's used deeper in the actually tree (since splitting the data could make the feature get better
        # here ensemble tree could help -> recalc the samples + feature again after split first tree?
        # the feature it choses will be the best split.
        all_splits = self.remove_duplicate_splits(all_splits)
        keys = np.array(list(all_splits.keys()))
        possible_splits = np.array(list(all_splits.values()))
        good_splits_dict = {}
        while len(good_splits_dict) < n_splits and possible_splits.shape[0] > 0:
            pred = Predictor(possible_splits.T, self.feature_to_samples(), [self.feature_type],
                             max_depth=1, min_sup=1,time=100)
            tree = pred.dl_predictor.tree_
            print(tree.keys())
            try:
                split_id = tree['feat']
                total_error = tree['left']['error'] + tree['right']['error']
            except KeyError:
                split_id = 0
                total_error = tree['error']
            print(f'found split: {keys[split_id]}')
            good_splits_dict[keys[split_id]] = possible_splits[split_id]
            self.errors.append(total_error)
            self.active_splits[keys[split_id]] = possible_splits[split_id]
            possible_splits = np.delete(possible_splits, split_id, axis=0)
            keys = np.delete(keys, split_id, axis=0)

    def remove_duplicate_splits(self, new_pos_splits: Dict[str, list]):
        #no need for finding the same splits (makes the model not perfom better in any way
        # exemple where this can happen with -> bool split on >5 and bool split on interval [0, 5] is the same
        splits_to_keep = {}
        new_vals = list(new_pos_splits.values())
        new_keys = list(new_pos_splits.keys())
        found_splits = list(self.active_splits.values())
        for i in range(len(new_keys)):
            if check_if_sub_array(found_splits, new_vals[i]):
                print("!Values is already present in the array!")
            else:
                splits_to_keep[new_keys[i]] = new_vals[i]
        return splits_to_keep


##################################################"
##################"HELPERS##########################"
####################################################
def standardize_feature(raw_feature_data):
    #standerazation is scaling down the feature to [0,1] values (it should not change relative gaps and shuch)
    #perhaps is scaling to [1,2] better, need to think about it
    if raw_feature_data is None:
        return None
    try:
        num_arr = np.asarray(raw_feature_data, dtype=float)
        print("Converted numeric strings to float.")
        return num_arr
    except (ValueError, TypeError):
        num_arr = value_to_index(raw_feature_data)
        print("Converted chars to index")
        return num_arr

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


