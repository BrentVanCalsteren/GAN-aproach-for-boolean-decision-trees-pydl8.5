##########################
### BIGGER
################################
import numpy as np

import CONFIG
from splits.split_type import SplitType
from usePydl.predictors.helpers.interval import Interval


class BiggerEqSplit(SplitType):

    def __init__(self, infos, index):
        super().__init__()
        self.vals, self.feats, self.splits, self.scores = None, None, None,None
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
        if sample[self.feats] >= self.vals:
            return 'L'
        return 'R'

    def left_interval(self):
        feat = int(self.feats)
        return {  feat: [Interval(self.vals, CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max[self.feats], "closed")]}

    def right_interval(self):
        feat = int(self.feats)
        return {  feat:  [Interval(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min[self.feats], self.vals, "half-closed")]}

    def to_string(self) -> str:
        return f"bigger_eq_{self.threshold}"

    def map_samples_to_split(self, samples: np.ndarray):
         return samples.T[self.feats] >= self.vals


    @classmethod
    def generate_splits(cls, candidates, thresholds, features: np.ndarray):
        for i, feat in enumerate(features):
            for val in thresholds[i]:
                splits = (feat >= val)
                cls.add_split(candidates, i, splits, val)