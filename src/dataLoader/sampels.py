from typing import List

import numpy as np
import src.dataLoader.dataset as loader
from dataLoader.feature import Feature

class Samples:
    def __init__(self,dataset='iris'):
        self.feature_list : List[Feature] = []
        self.samples_raw_complete = np.array([])
        self.samples_raw_missing = np.array([])
        self.samples = np.array([])
        if dataset:
            self.load_new_dataset(dataset)

    def load_new_dataset(self,dataset='iris'):
        loaded_data = loader.load_dataloader_by_name(dataset)
        self.samples_raw_complete = loaded_data.get_x_complete()
        np.random.shuffle(self.samples_raw_complete)
        self.samples_raw_missing = loaded_data.get_x_missing()
        self.feature_list = [Feature(feat) for feat in self.samples_raw_complete.T]

    def get_samples(self):
        return np.array([feat.feature_array for feat in self.feature_list]).T

    def find_best_splits(self):
        all_splits = []
        split_len = []

        def add_splits(best_splits, total_len):
            if best_splits:
                total_len += np.array(list(best_splits.values())).shape[0]
                all_splits.append(np.array(list(best_splits.values())))
            return total_len

        for feat in self.feature_list:
            total_len = 0
            best_splits, errors = feat.get_single_value_splits(10)
            total_len += add_splits(best_splits, total_len)

            best_splits, errors = feat.get_percentile_binning(10)
            total_len += add_splits(best_splits, total_len)

            best_splits, errors = feat.get_cluster_interval_splits(10)
            total_len += add_splits(best_splits, total_len)

            split_len.append(total_len)

        all_splits = np.vstack(all_splits).T
        return all_splits, split_len

    def map_other_samples_to_same_splits(self,other_samples : np.ndarray):
        if not self.feature_list:
            print('can map samples on same splits, list is empty')
        features = other_samples.T
        all_splits = []
        for i,feat in enumerate(self.feature_list):
            all_splits.append(feat.map_to_splits(features[i]))
        all_splits = np.vstack(all_splits).T
        return all_splits




