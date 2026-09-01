import re
from typing import List

import CONFIG
from src.usePydl.error_fun import IntervalSizesError, MSEError, ClusterCoherenceError
import numpy as np
from src.data.find_largest_gaps import DivisiveCluster
from scipy.stats import gaussian_kde
from src.data.data_obj.splits.split_type import SplitType
from src.data.data_obj.splits.bigger_split import BiggerEqSplit
from src.data.data_obj.splits.equal_split import EqualSplit
from src.data.data_obj.splits.interval_split import IntervalSplit

KEYS = ['types', 'values', 'feats', 'splits']

class Splits:
    splits_dic = None
    values = None
    feature_index_array:List = None
    def __init__(self, features, max_splits_each_feature:None, weights_each_feature=None):
        samples = features.T
        self.max_splits_each_feature = max_splits_each_feature
        self.weights_each_feature = weights_each_feature
        self.splits_dic = {}
        self.interval_error = IntervalSizesError(samples=samples,feature_weights=weights_each_feature)
        self.mse_error = MSEError(samples=samples,feature_weights=weights_each_feature)
        self.cluster_error = ClusterCoherenceError(samples=samples,feature_weights=weights_each_feature)
        self.create_best_splits(features)

    def get_splits_array(self):
        return np.array(self.splits_dic['splits'])

    def get_data_on_split_id(self, bool_feat_id):
        data = []
        for key in KEYS:
            data.append(self.splits_dic[key][bool_feat_id])
        return data


    def create_best_splits(self, features):
        all_candidates = _generate_splits(features)
        all_splits = np.array(all_candidates['splits'])
        scores = self.score_splits(all_splits)
        best_indices = np.argsort(-scores)[:CONFIG.MAX_SPLITS]
        best_cand = SplitType.create_empty_split_directory()
        for key in KEYS:
            val = all_candidates[key]
            best_cand[key] = [val[i] for i in best_indices]
        best_cand['scores'] = scores[best_indices]
        #initing the split_classes
        for i in range(len(best_cand['types'])):
            best_cand['types'][i] = best_cand['types'][i](best_cand, i)
            best_cand['left_intervals'].append(best_cand['types'][i].left_interval())
            best_cand['right_intervals'].append(best_cand['types'][i].right_interval())
        self.splits_dic = best_cand



    def score_splits(self, splits):
        vals = np.unique(splits)
        is_0 = (splits == vals[0])

        def gini():
            prob_0 = np.mean(is_0, axis=1)
            prob_1 = 1.0 - prob_0
            return 1 - np.abs(prob_0 - prob_1)

        gini_score = gini()

        def interval_score():
            scores = np.array([self.interval_error(np.where(row_mask)[0]) + self.interval_error(np.where(~row_mask)[0])for row_mask in is_0])
            max_s = np.max(scores) if np.max(scores) > 0 else 1.0
            return 1 - scores / max_s

        inter_scores = interval_score()

        def mse_score():
            scores = np.array([self.mse_error(np.where(row_mask)[0]) + self.mse_error(np.where(~row_mask)[0]) for row_mask in is_0])
            max_s = np.max(scores) if np.max(scores) > 0 else 1.0
            return 1 - scores / max_s

        mse_score = mse_score()

        def cluster_score():
            valid_mask = np.array([np.sum(m) >= 2 and np.sum(~m) >= 2 for m in is_0])
            scores = np.full(len(is_0), np.nan)

            for idx in np.where(valid_mask)[0]:
                m = is_0[idx]
                scores[idx] = self.cluster_error(np.where(m)[0]) + self.cluster_error(np.where(~m)[0])

            valid_scores = scores[valid_mask]
            if len(valid_scores) > 0 and np.ptp(valid_scores) > 0:
                # Min-max scale valid splits to [0, 1] where lower error -> higher score
                scores[valid_mask] = 1.0 - (valid_scores - np.min(valid_scores)) / np.ptp(valid_scores)
            scores[~valid_mask] = 0.0
            return scores

        cluster_score = cluster_score()

        return 0.5*cluster_score + 0.15*gini_score + 0.5*inter_scores + 0.35*mse_score


    def map_samples_to_splits(self, samples):
        if self.splits_dic is None:
            print('cant map splits, since its none')
            return None
        split_classes = self.splits_dic['types']

        new_splits = np.zeros(((len(split_classes)), samples.shape[0]))
        for i, split in enumerate(split_classes):
            new_splits[i] = np.array(split.map_samples_to_split(samples))

        return new_splits


###############################"
######helpers


def _generate_splits(features):
    candidates = SplitType.create_empty_split_directory()
    all_thresholds = calc_all_cont_tresholds(features)
    BiggerEqSplit.generate_splits(candidates,all_thresholds,features)
    EqualSplit.generate_splits(candidates,all_thresholds,features)
    IntervalSplit.generate_splits(candidates,all_thresholds,features)
    return candidates


def calc_all_cont_tresholds(features):
    tresh_holds = {}
    for i, feature in enumerate(features):
        # Percentile binning
        percentiles = np.linspace(5, 95, CONFIG.TRESHHOLD_GRAIN)
        p_thresholds = np.unique(np.percentile(feature, percentiles))

        # Density Valleys (KDE local minima)
        valley_thresholds = find_kde_valleys(feature, num_points=feature.size)

        # Clustering Thresholds
        cluster_thresholds = []
        try:
            cluster = DivisiveCluster()
            cluster.max_depth = CONFIG.TRESHHOLD_GRAIN
            cluster.fit(feature)
            for k in range(1, CONFIG.TRESHHOLD_GRAIN):
                intervals = cluster.get_clusters_at_depth(k)
                for interval in intervals:
                    cluster_thresholds.extend([interval[0], interval[1]])
        except Exception:
            pass

        tresh_holds[i] = np.unique(np.concatenate((p_thresholds, valley_thresholds, cluster_thresholds)))  # remove dubles
    return tresh_holds






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