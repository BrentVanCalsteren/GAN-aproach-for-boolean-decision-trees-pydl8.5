import numpy as np
import CONFIG

from data.data_loader.dataLoader import load_dataloader_by_name
from data.data_obj.chunk_info import GlobalChunkInfo
from data.data_obj.feature_history import FeatureHistory

class Samples:
    def __init__(self, dataset: str = 'iris', data_type='tabular'):
        self.current_feat_hist: FeatureHistory = None
        self.chunk_info: GlobalChunkInfo = None
        self.copy_hist: FeatureHistory = None
        self.samples = np.array([])
        self.labels = np.array([])
        self.loader = None

        self.load_new_dataset(dataset, data_type)

    def load_new_dataset(self, dataset='iris', data_type='tabular'):
        loader = load_dataloader_by_name(dataset_name=dataset, data_type=data_type)
        self.loader = loader
        self.chunk_info = GlobalChunkInfo(loader=loader)
        self.update_pre_process_global_vals()

    def update_pre_process_global_vals(self):
        n_chunks = self.loader.n_chunks
        max_vals = np.full(len(self.chunk_info.featureTypes), -np.inf)
        min_vals = np.full(len(self.chunk_info.featureTypes), np.inf)
        for chunk_id in range(n_chunks):
            self.load_chunk(chunk_id)
            for i in range(self.samples.shape[1]):
                col = self.samples[:, i]
                min_vals[i] = min(min_vals[i], float(np.min(col)))
                max_vals[i] = max(max_vals[i], float(np.max(col)))
        self.chunk_info.processed_feat_max = max_vals
        self.chunk_info.processed_feat_min = min_vals

    def load_chunk(self, chunk_id):
        samples, labels =  self.loader.load_chunk(chunk_id)
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        samples = samples[indices]  # shuffle
        labels = labels[indices]  # shuffle
        self.samples = samples
        self.labels = labels
        self.pre_process_samples()

    def get_best_matching_label(self, samples, chunk_id):
        known_labels = self.labels.flatten()
        labels = []
        for sample in samples:
            distances = np.linalg.norm(self.samples - sample, axis=1)
            closest_index = np.argmin(distances)
            labels.append(known_labels[closest_index])
        return np.array(labels)

    def save_feature_history(self, is_scale):
        new_hist = FeatureHistory(samples=self.samples, chunkInfo=self.chunk_info, is_scale=is_scale)
        new_hist.past = self.current_feat_hist
        self.copy_hist = new_hist
        self.current_feat_hist = new_hist
        self.samples = new_hist.get_sample_array_from_history()


    def reverse_history(self,samples, is_scale):
        if is_scale:
            old_samples = self.copy_hist.get_rescaled_sample_based_on_history(samples)
        else: old_samples = samples
        self.copy_hist = self.copy_hist.past
        return old_samples

    def reset_copy_hist(self):
        self.copy_hist = self.current_feat_hist

#####################################################################################
#################### process samples functions #######################################

    def pre_process_samples(self):
        self.save_feature_history(is_scale=True)
        if CONFIG.ROTATE_DIM:
            self.aply_PCA_rotation()
        if CONFIG.REDUCE_FEAT:
            self.reduce_with_NN()
        self.save_feature_history(is_scale=False)


    def reverse_process_samples(self, samples):
        labels = self.labels
        if samples is None:
            samples = self.samples
        samples = self.reverse_history(samples, is_scale=False)
        if CONFIG.REDUCE_FEAT:
            samples = self.extend_with_NN(samples)
        if CONFIG.ROTATE_DIM:
            samples = self.restore_PCA(samples)
        samples = self.reverse_history(samples, is_scale=True)
        self.reset_copy_hist()
        return samples, labels.flatten()
############ helper #######################

    def aply_PCA_rotation(self):
        self.samples = self.chunk_info.global_pca.transform(self.samples)

    def restore_PCA(self, samples):
        return self.chunk_info.global_pca.inverse_transform(samples)

    def reduce_with_NN(self):
        self.samples = self.chunk_info.global_nn.transform(self.samples)

    def extend_with_NN(self, samples):
        return self.chunk_info.global_nn.inverse_transform(samples)

##################### END ###############################

    def get_samples(self):
        return self.samples

    def save_output(self,samples, llables, output_name):
        samples, labels = self.reverse_process_samples(samples)
        if llables is not None:
            labels = np.array(llables)
        self.loader.save_output_to_folder(samples,labels, filename=output_name)


#################
#other functions
#################
def value_to_index(array):
    unique_values = np.unique(array)
    indeces = np.zeros(array.shape)
    for i, val in enumerate(unique_values):
        indeces[array == val] = i
    return indeces
