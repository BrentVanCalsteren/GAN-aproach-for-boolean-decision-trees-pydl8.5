import re
from typing import Dict, List
from src.usePydl.error_fun import IntervalSizesError, MSEError
import numpy as np
from src.data.find_largest_gaps import DivisiveCluster
from scipy.stats import gaussian_kde


class Splits:
    splits = None
    values = None
    feature_index_array:List = None
    def __init__(self, max_splits_each_feature, samples, weights=None):
        self.max_splits_each_feature = max_splits_each_feature
        self.splits = []
        self.values = []
        self.samples = samples
        self.weights = weights
        self.interval_error = IntervalSizesError(samples=self.samples,feature_weights=self.weights)
        self.mse_error = MSEError(samples=self.samples,feature_weights=self.weights)

    def get_splits(self, cut=None):
        if cut is not None and cut == -1:
            m = np.max(self.feature_index_array)
            mask = (self.feature_index_array != m)
            splits = np.array(self.splits)[mask]
        else:
            splits = np.array(self.splits)
        return splits.T


    def create_splits_from_feature(self, feature_array: np.ndarray,feat_id):
        #print('generating splits for feature')
        new_splits = self._generate_best_splits(feature_array, feat_id)
        n_new_splits = self._save_best_splits(new_splits, feat_id)
        if self.feature_index_array is None:
            self.feature_index_array = [0]*n_new_splits
        else:
            max = np.max(self.feature_index_array)
            self.feature_index_array += [max+1]*n_new_splits

    def _save_best_splits(self, new_splits,feat_id):
        splits = np.array(list(new_splits.values()))
        scores = self.score_splits(splits)
        vals = list(new_splits.keys())
        splits = np.array(list(new_splits.values()))
        valid_candidates = list(zip(scores, vals, splits))
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidates = valid_candidates[:self.max_splits_each_feature[feat_id]]
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
            interval_scores = np.array([self.interval_error(np.where(row_mask)[0]) + self.interval_error(np.where(~row_mask)[0])for row_mask in is_0])
            max_s = np.max(interval_scores) if np.max(interval_scores) > 0 else 1.0
            return 1 - interval_scores / max_s

        inter_scores = interval_score()

        def mse_score():
            interval_scores = np.array([self.mse_error(np.where(row_mask)[0]) + self.mse_error(np.where(~row_mask)[0]) for row_mask in is_0])
            max_s = np.max(interval_scores) if np.max(interval_scores) > 0 else 1.0
            return 1 - interval_scores / max_s

        mse_score = mse_score()

        return 0.15*gini_score + 0.5*inter_scores + 0.35*mse_score

    def _generate_best_splits(self, feature, feat_id):
        new_splits = self._generate_discrete_splits(feature,feat_id)
        new_splits.update(self._generate_continues_splits(feature, feat_id))
        for i in range(self.samples.shape[1]):
            if i != feat_id:
                continue #skipping this, just proof of concept
                new_splits.update(self.generate_linear_combination_splits(feat_id, i, max_splits=self.max_splits_each_feature[feat_id]))
        return remove_duplicate_splits(new_splits)

    def _generate_discrete_splits(self, feature,feat_id):
        new_splits = {}
        uniques, counts = np.unique(feature, return_counts=True)
        top_uniques = uniques[np.argsort(-counts)][:self.max_splits_each_feature[feat_id]]
        for val in top_uniques:
            new_splits[f"even_{val}"] = (feature == val).astype(int)
        return new_splits

    def _generate_continues_splits(self, feature, feat_id):
        new_splits = {}
        max_s = self.max_splits_each_feature[feat_id]

        #Percentile binning
        percentiles = np.linspace(5, 95, max_s * 2)
        p_thresholds = np.unique(np.percentile(feature, percentiles))

        #Density Valleys (KDE local minima)
        valley_thresholds = find_kde_valleys(feature,num_points=feature.size)

        #Clustering Thresholds
        cluster_thresholds = []
        try:
            cluster = DivisiveCluster()
            cluster.max_depth = max_s*2
            cluster.fit(feature)
            intervals = cluster.get_clusters_at_depth(max_s*2)
            for interval in intervals:
                cluster_thresholds.extend([interval[0], interval[1]])
        except Exception:
            pass

        all_thresholds = np.unique(np.concatenate((p_thresholds, valley_thresholds,cluster_thresholds)))

        #gen candidates based on thresholds
        candidates = []
        for t in all_thresholds:
            candidates.append(('bigger_eq', t, (feature >= t)))
            candidates.append(('smaller_eq', t, (feature <= t)))

        for i in range(len(all_thresholds) - 1):
            for j in range(i + 1, min(i + 4, len(all_thresholds))):
                t1, t2 = all_thresholds[i], all_thresholds[j]
                candidates.append(('interval', (t1, t2), (feature >= t1) & (feature <= t2)))

        for c_type, t_val, mask in candidates:
            if c_type == 'interval':
                new_splits[f"interval_{t_val[0]}_{t_val[1]}"] = mask.astype(int)
            else:
                new_splits[f"{c_type}_{t_val}"] = mask.astype(int)
        return new_splits

    def generate_linear_combination_splits(self, feat_id_1: int, feat_id_2: int, max_splits: int = 4):
        feature_i = self.samples.T[feat_id_1]
        feature_j = self.samples.T[feat_id_2]
        new_splits = {}

        angles = [(0.7071, 0.7071), (0.7071, -0.7071)]  #45 deg and 135 deg
        for w1, w2 in angles:
            z = w1 * feature_i + w2 * feature_j
            if np.all(z == z[0]):
                continue
            percentiles = np.linspace(20, 80, min(max_splits, 5))
            thresholds = np.unique(np.percentile(z, percentiles))
            for t in thresholds:
                key = f"lincomp_{feat_id_1}_{feat_id_2}_{w1}_{w2}_>=_{t}"
                new_splits[key] = (z >= t).astype(int)
        return new_splits

    def map_samples_to_splits(self, samples):
        features = samples.T
        new_splits = np.zeros((self.get_splits().shape[1], samples.shape[0]))
        for i, val_str in enumerate(self.values):
            feature_id = self.feature_index_array[i]
            if 'lincomp' in val_str:
                parts = val_str.split('_')
                f1, f2 = int(parts[1]), int(parts[2])
                w1, w2 = float(parts[3]), float(parts[4])
                t = float(parts[6])
                z = w1 * features[f1] + w2 * features[f2]
                new_splits[i] = np.where(z >= t, 1, 0)
            else:
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
        pass
        #print(f"Removed {removed_count} duplicate/inverse splits.")

    return splits_to_keep


def find_kde_valleys(feature: np.ndarray, num_points: int = 100) -> np.ndarray:
    if len(feature) < 10 or np.all(feature == feature[0]): return np.array([])
    try:
        grid = np.linspace(np.min(feature), np.max(feature), max(100, num_points))
        kde = gaussian_kde(feature, bw_method=0.15)
        density = kde(grid)
        minima_indices = (density[1:-1] < density[:-2]) & (density[1:-1] < density[2:])
        valleys = grid[1:-1][minima_indices]

        if len(valleys) == 0:
            kde_alt = gaussian_kde(feature, bw_method=0.3)
            density_alt = kde_alt(grid)
            minima_indices_alt = (density_alt[1:-1] < density_alt[:-2]) & (density_alt[1:-1] < density_alt[2:])
            valleys = grid[1:-1][minima_indices_alt]
        return valleys
    except Exception:
        return np.array([])