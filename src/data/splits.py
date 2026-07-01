import re
from typing import Dict, List
from src.usePydl.error_fun import IntervalSizesError

import numpy as np

from data import feature
from data.feature import Feature
from src.data.divisive_clustering_1D import DivisiveCluster



class Splits:
    splits = None
    values = None
    feature_index_array:List = None
    def __init__(self, max_splits_each_feature, sample_obj):
        self.max_splits_each_feature = max_splits_each_feature
        self.splits = []
        self.values = []
        self.sample_obj = sample_obj

    def get_splits(self, cut=None):
        if cut is not None and cut == -1:
            m = np.max(self.feature_index_array)
            mask = (self.feature_index_array != m)
            splits = np.array(self.splits)[mask]
        else:
            splits = np.array(self.splits)
        return splits.T


    def create_splits_from_feature(self, feature: Feature):
        new_splits = self._generate_best_splits(feature)
        n_new_splits = self._save_best_splits(new_splits)
        if self.feature_index_array is None:
            self.feature_index_array = [0]*n_new_splits
        else:
            max = np.max(self.feature_index_array)
            self.feature_index_array += [max+1]*n_new_splits

    def _save_best_splits(self, new_splits):
        valid_candidates = []
        splits = np.array(list(new_splits.values()))
        scores = self.score_splits(splits)
        vals = list(new_splits.keys())
        splits = np.array(list(new_splits.values()))
        valid_candidates = list(zip(scores, vals, splits))
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidates = valid_candidates[:self.max_splits_each_feature]
        new_best_splits = [c[2] for c in best_candidates]
        new_best_vals = [c[1] for c in best_candidates]
        self.splits.extend(new_best_splits)
        self.values.extend(new_best_vals)
        return len(new_best_splits)

    def score_splits(self, splits):
        vals = np.unique(splits)
        is_0 = (splits == vals[0])

        def gini():
            prob_0 = np.mean(is_0, axis=1)
            prob_1 = 1.0 - prob_0
            return 1 - np.abs(prob_0 - prob_1)

        gini_score = gini()

        def interval_score():
            error_calc = IntervalSizesError(samples=self.sample_obj.get_samples())
            interval_scores = np.array([error_calc(np.where(row_mask)[0]) + error_calc(np.where(~row_mask)[0])for row_mask in is_0])
            return 1- interval_scores/np.max(interval_scores)

        inter_scores = interval_score()
        return gini_score + inter_scores

    def _generate_best_splits(self, feature):
        if feature.isDiscrete:
            new_splits = self._generate_discrete_splits(feature)
        else:
            new_splits = self._generate_continues_splits(feature)
        return remove_duplicate_splits(new_splits)

    def _generate_discrete_splits(self, feature):
        new_splits = {}
        uniques, counts = np.unique(feature.feature_array, return_counts=True)
        top_uniques = uniques[np.argsort(-counts)]
        for val in top_uniques:
            new_splits[f"even_{val}"] = (feature.feature_array == val).astype(int)
        return new_splits

    def _generate_continues_splits(self, feature):
        new_splits = {}
        feature_array =  feature.feature_array
        #binning
        thresholds = None
        for i in range(feature_array.shape[0]):
            percentiles = np.linspace(5, 95, min(self.max_splits_each_feature * 2, 20))
            if thresholds is None:
                thresholds = np.unique(np.percentile(feature_array, percentiles))
            else:
                thresholds = np.unique(np.concatenate((thresholds, np.percentile(feature_array, percentiles))))

        #clustering
        cluster_thresholds = []
        cluster = DivisiveCluster()
        cluster.max_depth = self.max_splits_each_feature
        cluster.fit(feature_array)
        for i in range(feature_array.shape[0]):
            intervals = cluster.get_clusters_at_depth(i)
            for interval in intervals:
                cluster_thresholds.extend([interval[0], interval[1]])

        #generate candidates based on tresholds
        candidates = []
        thresholds = np.unique(np.concatenate((thresholds, cluster_thresholds)))
        for t in thresholds:
            candidates.append(('bigger_eq', t, (feature_array >= t)))
            candidates.append(('smaller_eq', t, (feature_array <= t)))

        for i in range(len(thresholds) - 1):
            for j in range(i + 1, min(i + 5, len(thresholds))):
                t1, t2 = thresholds[i], thresholds[j]
                candidates.append(('interval', (t1, t2), (feature_array >= t1) & (feature_array <= t2)))

        for c_type, t_val, mask in candidates:
            if c_type == 'interval':
                new_splits[f"interval_{t_val[0]}_{t_val[1]}"] = mask.astype(int)
            else:
                new_splits[f"{c_type}_{t_val}"] = mask.astype(int)
        return new_splits

    def map_samples_to_splits(self, samples, cut=None):
        features = samples.T
        new_splits = np.zeros((self.get_splits().shape[1], samples.shape[0]))
        for i, val_str in enumerate(self.values):
            feature_id = self.feature_index_array[i]
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            vals = [float(x) for x in matches]
            if 'bigger_eq' in val_str:
                new_splits[i] = np.where(features[feature_id] >= vals[0], 1, 0)
            elif 'even' in val_str:
                new_splits[i] = np.where(features[feature_id] == vals[0], 1, 0)
            elif 'smaller_eq' in val_str:
                new_splits[i] = np.where(features[feature_id] <= vals[0], 1, 0)
            elif 'interval' in val_str:
                new_splits[i] = np.where((features[feature_id] >= vals[0]) & (features[feature_id] <= vals[1]), 1, 0)
            else:
                raise ValueError('invalid key')
        if cut is not None and cut == -1:
            m = np.max(self.feature_index_array)
            mask = (self.feature_index_array != m)
            new_splits = new_splits[mask]
        return new_splits.T


###############################"
######helpers

def remove_duplicate_splits(new_pos_splits: Dict[str, list]):
    if not new_pos_splits:
        return {}

    keys = list(new_pos_splits.keys())
    vals = np.array(list(new_pos_splits.values()))

    if vals.size == 0:
        return new_pos_splits
    first_elements = vals[:, 0]
    normalized_vals = np.where(first_elements[:, None] == 1, 1 - vals, vals)
    _, unique_indices = np.unique(normalized_vals, axis=0, return_index=True)


    unique_indices.sort()
    splits_to_keep = {str(keys[i]): vals[i] for i in unique_indices}
    removed_count = len(keys) - len(unique_indices)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate/inverse splits.")

    return splits_to_keep