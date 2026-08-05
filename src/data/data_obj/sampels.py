import random

import numpy as np
from sklearn.model_selection import train_test_split

import CONFIG

from data.data_loader.dataLoader import load_dataloader_by_name
from data.data_obj.chunk_info import GlobalChunkInfo
from data.data_obj.feature_history import FeatureHistory

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
        self.active_chunk = chunk_id
        self.current_feat_hist = None
        samples, lables = self.loader.load_chunk(chunk_id, self.labels_at_front)
        samples_train, samples_test, labels_train, labels_test = train_test(samples=samples, labels=lables, test_size=split_test)
        self.samples = self.pre_process_samples(samples=samples_train)
        self.save_feature_history(samples=self.samples)
        self.samples_test = self.pre_process_samples(samples=samples_test)
        self.labels = labels_train
        self.labels_test = labels_test


    def get_best_matching_label(self, samples, chunk_id=None):
        if chunk_id is not None and chunk_id != self.active_chunk:
            self.load_chunk(chunk_id)
        known_labels = self.labels.flatten()
        if len(known_labels) == 0 or self.samples.size == 0:
            raise ValueError("No reference samples or labels available in Samples object.")
        if self.chunk_info.feature_importance is not None:
            weights = np.asarray(self.chunk_info.feature_importance, dtype=float)
            if weights.shape[0] == samples.shape[1]:
                weights_sqrt = np.sqrt(np.maximum(weights, 0.0))
                diff = (samples[:, np.newaxis, :] - self.samples[np.newaxis, :, :]) * weights_sqrt
                distances = np.linalg.norm(diff, axis=2)
            else:
                diff = samples[:, np.newaxis, :] - self.samples[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=2)
        else:
            diff = samples[:, np.newaxis, :] - self.samples[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
        closest_indices = np.argmin(distances, axis=1)
        return known_labels[closest_indices]

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