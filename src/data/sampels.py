from typing import List, Optional

import numpy as np
from PIL import features

from data.dataLoader import load_dataloader_by_name as loader, array_to_image, save_image_to_folder
from data.feature import Feature
import data.feature as f

class Samples:
    def __init__(self,dataset : Optional[str]='iris',data_type='tabular',set_all_discrete=False):
        if set_all_discrete:
            f.DESCRETE_PERCENTILE = 100
        self.feature_list : List[Feature] = []
        self.samples = np.array([])
        self.encoder = None
        if dataset:
            self.load_new_dataset(dataset,data_type)

    def load_new_dataset(self,dataset='iris',data_type='tabular'):
        loaded_data = loader(dataset_name=dataset, data_type=data_type)
        samples = loaded_data.get_x_complete()
        self.encoder = loaded_data.encoder
        self.load_samples(samples)


    def load_samples(self, samples:np.ndarray):
        np.random.shuffle(samples)
        self.samples = samples
        self.feature_list = [Feature(feat) for feat in self.samples.T]

    def get_samples(self, slices: Optional[slice] = None, convert_to_int=False):
        if slices is None:
            sliced = self.feature_list
        else:
            sliced = self.feature_list[slices]
        if len(sliced) == 1:
            if convert_to_int:
                return np.array([value_to_index(feat.feature_array) for feat in sliced]).flatten()
            else:
                return np.array([feat.feature_array for feat in sliced]).flatten()
        return np.array([feat.feature_array for feat in sliced]).T

    def creat_splits(self, splits_each_feature=None):
        if splits_each_feature is None:
            splits_each_feature = 5
        for feat in self.feature_list:
            feat.create_all_independent_splits(splits_each_feature)

    def get_splits(self, slices: Optional[slice] = None):
        if slices is None:
            sliced = self.feature_list
        else:
            sliced = self.feature_list[slices]
        all_bool_splits = np.array([])
        for feat in sliced:
            splits = feat.get_splits_as_array()
            if all_bool_splits.size > 0:
                all_bool_splits = np.vstack((all_bool_splits, splits))
            else:
                all_bool_splits = splits
        return np.array(all_bool_splits).T

    def map_other_samples_to_same_splits(self,other_samples : np.ndarray, slices: Optional[slice] = None):
        if not self.feature_list:
            print('cant map samples on same splits, list is empty')
            return None
        features = other_samples.T
        if slices is None:
            sliced_features = features
            feat_list_sliced = self.feature_list
        else:
            sliced_features = features[slices]
            feat_list_sliced = self.feature_list[slices]
        all_splits = []
        for i,feat in enumerate(feat_list_sliced):
            all_splits.append(feat.map_feature_to_active_splits(sliced_features[i]))
        all_splits = np.vstack(all_splits).T
        return all_splits

    def get_feature_types(self):
        return [feat.feature_type for feat in self.feature_list]

    def reverse_scale(self, s):
        fifa = s.T
        reverse = []
        for i, feat in enumerate(self.feature_list):
            reverse.append(feat.reverse_scale(fifa[i]))
        return np.array(reverse).T

    def decode_samples(self, samples_to_decode):
        if self.encoder:
            decoded = self.encoder.inverse_transform(samples_to_decode)
            return decoded
        print("can't decode, no encoder")
        return samples_to_decode

    def convert_to_image(self,samples, name='default.png'):
        reverse_scaled = self.reverse_scale(samples)
        decoded_samples = self.decode_samples(reverse_scaled[:10, :-1])
        image = array_to_image(decoded_samples[0])
        if not '.' in name:
            name = name + '.png'
        save_image_to_folder(image, "output", name)

#helpers
def value_to_index(array):
    unique_values = np.unique(array)
    indeces = np.zeros(array.shape)
    for i, val in enumerate(unique_values):
        indeces[array==val] = i
    return indeces


