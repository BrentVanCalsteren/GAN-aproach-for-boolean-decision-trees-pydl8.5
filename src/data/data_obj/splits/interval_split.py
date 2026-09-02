from abc import ABC

import numpy as np

import CONFIG
from splits.split_type import SplitType
from usePydl.predictors.helpers.interval import Intervals, Interval
from itertools import combinations_with_replacement


#########################################
########### INTERVAL
###################################"


class IntervalSplit(SplitType):

    def __init__(self, infos, index):
        super().__init__()
        self.vals, self.feats, self.splits, self.scores = None, None, None, None
        for key in ['values', 'feats', 'splits', 'scores']:
            if key == 'values':
                self.vals = np.array(infos[key][index])
            elif key == 'feats':
                self.feats = np.array(infos[key][index])
            elif key == 'splits':
                self.splits = np.array(infos[key][index])
            elif key == 'scores':
                self.scores = np.array(infos[key][index])

    def evaluate_sample(self, sample):
        if  (sample[self.feats] >= self.vals[0]) & (sample[self.feats] <= self.vals[1]):
            return 'L'
        return 'R'

    def left_interval(self):
        feat = int(self.feats)
        return {  feat:[Interval(self.vals[0], self.vals[1], "closed")]}

    def right_interval(self):
        feat = int(self.feats)
        return {  feat:[
            Interval(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min[self.feats], self.vals[0], "half-closed"),
            Interval(self.vals[1], CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max[self.feats], "half-open")
        ]}


    def map_samples_to_split(self, samples: np.ndarray):
        return (samples.T[self.feats] >= self.vals[0]) & (samples.T[self.feats] <= self.vals[1])

    def to_string(self) -> str:
        return f"interval_{self.min_val}_{self.max_val}"

    @classmethod
    def generate_splits(cls, candidates, thresholds, features: np.ndarray):
        for i, feat in enumerate(features):
            thresh = np.unique(thresholds[i])
            if len(thresh) == 0:
                continue

            pairs = np.array(list(combinations_with_replacement(thresh, 2)))
            lows = pairs[:, 0]
            highs = pairs[:, 1]
            feat_row = feat[None, :]
            splits_matrix = (feat_row >= lows[:, None]) & (feat_row <= highs[:, None])

            # 3. Add all splits for this feature in a single batch
            cls.add_splits(candidates, i, splits_matrix, pairs)