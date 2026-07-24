from typing import Optional
import numpy as np

from data.data_loader.dataLoader import load_dataloader_by_name
from data.data_obj.splits import Splits
from src.data.encoders.encoder_PCA import PCAEncoder
from src.data.encoders.encoder_NN import NNencoder
from data.data_obj.feature_history import FeatureHistory

ROTATE_DIM = True
USE_NN = False
REDUCE_FEAT = True
COMBINE_FEAT_LABELS = True

class Samples:
    def __init__(self, dataset: str = 'iris', data_type='tabular', set_all_discrete=False):
        self.current_feat_hist: FeatureHistory = None
        self.past_feat_hist: FeatureHistory = None
        self.splits_obj = None
        self.samples = np.array([])
        self.subsamples = list()
        self.labels = np.array([])
        self.loader = None
        self.pca = None
        self.nn = None
        self.load_new_dataset(dataset, data_type)
        self.pre_process_samples()

    def load_new_dataset(self, dataset='iris', data_type='tabular'):
        loader = load_dataloader_by_name(dataset_name=dataset, data_type=data_type)
        self.loader = loader
        samples, labels = loader.get_samples() #sampels and labels should already be procecced into numeric values + samples is a 2d np array
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        samples = samples[indices] #shuffle
        labels = labels[indices] #shuffle
        self.samples = samples
        self.labels = labels

    def save_feature_history(self):
        new_hist = FeatureHistory(samples=self.samples)
        new_hist.past = self.current_feat_hist
        self.current_feat_hist = new_hist
        self.past_feat_hist = new_hist
        self.samples = new_hist.get_sample_array_from_history()


    def reverse_history(self,samples):
        old_samples = self.past_feat_hist.get_rescaled_sample_based_on_history(samples)
        self.past_feat_hist = self.past_feat_hist.past
        return old_samples

#####################################################################################
#################### process samples functions #######################################

    def pre_process_samples(self):
        self.save_feature_history()
        if ROTATE_DIM:
            self.aply_PCA_rotation()
        if REDUCE_FEAT:
            self.reduce_with_NN()
        self.save_feature_history()


    def reverse_process_samples(self, samples):
        labels = self.labels
        if samples is None:
            samples = self.samples
        samples = self.reverse_history(samples)
        if REDUCE_FEAT:
            samples = self.extend_with_NN(samples)
        if ROTATE_DIM:
            samples = self.restore_PCA(samples)
        samples = self.reverse_history(samples)
        return samples, labels.flatten()
############ helper #######################

    def aply_PCA_rotation(self):
        self.pca = PCAEncoder(output_dim=min(self.samples.shape))
        self.samples = self.pca.transform(self.samples)

    def restore_PCA(self, samples):
        return self.pca.inverse_transform(samples)

    def reduce_with_NN(self, num_feats=50):
        if USE_NN:
            self.nn = NNencoder(output_dim=num_feats)
            self.nn.create_new_nn_module(samples=self.samples)
            self.nn.train_module(min_error=0.00005)
        else:
            self.nn = PCAEncoder(output_dim=num_feats)
        self.samples = self.nn.transform(self.samples)

    def extend_with_NN(self, samples):
        return self.nn.inverse_transform(samples)

##################### END ###############################

    def get_samples(self, slices: Optional[slice] = None, convert_to_int=False):
        if slices is None:
            sliced = self.samples
        else:
            sliced = self.samples[:, slices]
        if convert_to_int:
            sliced = sliced.copy()
            for i in range(sliced.shape[1]):
                sliced[:, i] = value_to_index(sliced[:, i])
        return sliced

    def creat_splits(self, max_num_splits_each_feature=None):
        if max_num_splits_each_feature is None:
            max_num_splits_each_feature = 5
        self.splits_obj = Splits(max_splits_each_feature=max_num_splits_each_feature, sample_obj=self)
        for feature_info in self.current_feat_hist.feature_info_list:
            self.splits_obj.create_splits_from_feature(feature_info.feature_array,feature_info.featureType)

    def get_splits_obj(self):
        return self.splits_obj

    def get_feature_info(self):
        return [feat.featureType for feat in self.feature_list]

    def save_output(self,samples, output_name):
        samples, labels = self.reverse_process_samples(samples)
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
