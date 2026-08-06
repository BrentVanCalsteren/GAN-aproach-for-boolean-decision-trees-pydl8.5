import gc

import numpy as np
from sklearn.model_selection import train_test_split

from data.data_loader.dataLoader import load_dataloader_by_name
from data.data_obj.chunk_info import GlobalChunkInfo
from data.data_obj.feature_history import FeatureHistory
from collections import Counter

class Samples:
    def __init__(self, dataset: str = 'iris', data_type='tabular', labels_at_front=False):
        self.current_feat_hist: FeatureHistory = None
        self.chunk_info: GlobalChunkInfo = None

        self.samples = np.array([])
        self.labels = np.array([])
        self.samples_test = np.array([])
        self.labels_test = np.array([])

        self.loader = None
        self.labels_at_front = labels_at_front
        self.active_chunk = None

        self.load_new_dataset(dataset, data_type)


    def load_new_dataset(self, dataset='iris', data_type='tabular'):
        loader = load_dataloader_by_name(dataset_name=dataset, data_type=data_type)
        self.loader = loader
        self.chunk_info = GlobalChunkInfo(loader=loader, labels_at_front=self.labels_at_front)


    def load_chunk(self, chunk_id, split_test=0.0):
        self.remove_memory()
        self.active_chunk = chunk_id
        samples, lables = self.loader.load_chunk(chunk_id, self.labels_at_front)
        samples = convert_samples_to_num(samples)
        samples_train, samples_test, labels_train, labels_test = train_test(samples=samples, labels=lables, test_size=split_test)
        self.samples = self.pre_process_samples(samples=samples_train)
        self.save_feature_history(samples=self.samples)
        self.samples_test = self.pre_process_samples(samples=samples_test)
        self.labels = labels_train
        self.labels_test = labels_test

    def remove_memory(self):
        if self.current_feat_hist is not None:
            self.current_feat_hist.reduce_memory()
        self.current_feat_hist = None
        self.samples = np.array([])
        self.samples_test = np.array([])
        gc.collect()

    def get_best_matching_labels(self, gen_samples: np.ndarray,chunk_id=0,k= 5):
        self.load_chunk(chunk_id)
        labels_flat = np.asarray(self.labels).flatten()
        real_samples = self.samples
        weights = self.chunk_info.feature_importance
        gen_labels = []
        for i, sample in enumerate(gen_samples):
            diff = np.abs(real_samples - sample)
            if weights is not None: diff = diff * weights
            distances = np.sum(diff, axis=1)
            best_ids = np.argpartition(distances, k)[:k]
            best_ids = best_ids[np.argsort(distances[best_ids])]
            possible_labels = labels_flat[best_ids]
            most_common_label = Counter(possible_labels).most_common(1)[0][0]
            gen_labels.append(most_common_label)
        return np.array(gen_labels)

    def save_feature_history(self, samples = None):
        if samples is None: samples = self.samples
        new_hist = FeatureHistory(samples=samples, chunkInfo=self.chunk_info)
        new_hist.past = self.current_feat_hist
        self.current_feat_hist = new_hist
        return new_hist.get_sample_array_from_history()


    def pre_process_samples(self, samples):
        if np.array(samples).shape[0] == 0: return samples
        return self.chunk_info.global_preprocessor.preprocess(samples)


    def reverse_process_samples(self, samples):
        if np.array(samples).shape[0] == 0: return samples
        return self.chunk_info.global_preprocessor.reverse_process(samples)

    def get_samples(self):
        return self.samples

    def save_output(self,samples, llables, output_name):
        samples = self.reverse_process_samples(samples)
        labels = self.labels
        if llables is not None:
            labels = np.array(llables)
        self.loader.save_output_to_folder(samples,labels, filename=output_name)


#################
#other functions
#################

def train_test(samples, labels, test_size=0.2):
    if test_size == 0.0: return samples, samples[:0], labels, labels[:0]
    if test_size == 1.0: return samples[:0], samples, labels[:0], labels
    return train_test_split(samples, labels, test_size=test_size, random_state=42)

def convert_samples_to_num(samples):
    num_feat = []
    featurs = samples.T
    for i, feat in enumerate(featurs):
        num_feat.append(make_num(feat))
    return np.array(num_feat).T.astype(float)


def make_num(raw_feature_data):
    try:
        num_arr = np.asarray(raw_feature_data, dtype=float)
    except (ValueError, TypeError):
        unique_values = np.unique(raw_feature_data)
        indexes = {val: idx for idx, val in enumerate(unique_values)}
        num_arr = np.array([indexes[val] for val in raw_feature_data]).astype(float)
        # maybe store the original strings? but would never need them (always want it to be numbers)
    return num_arr