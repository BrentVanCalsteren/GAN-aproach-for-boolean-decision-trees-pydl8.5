from typing import List, Optional
import numpy as np
from src.data.dataLoader import load_dataloader_by_name as loader, array_to_image, save_image_to_folder
from src.data.feature import Feature
from src.data.splits import Splits
import src.data.feature as f


class Samples:
    def __init__(self, dataset: Optional[str] = 'iris', data_type='tabular', set_all_discrete=False):
        if set_all_discrete:
            f.DESCRETE_PERCENTILE = 100
        self.feature_list: List[Feature] = []
        self.splits = None
        self.samples = np.array([])
        self.encoder = None
        if dataset is not None:
            self.load_new_dataset(dataset, data_type)

    def load_new_dataset(self, dataset='iris', data_type='tabular'):
        loaded_data = loader(dataset_name=dataset, data_type=data_type)
        samples = loaded_data.complete_X
        np.random.shuffle(samples)
        self.feature_list = [Feature(feat, self) for feat in samples.T]
        self.set_scaled_samples()
        self.encoder = loaded_data.encoder

    def set_scaled_samples(self):
        if self.feature_list:
            scaled = np.array([feat.feature_array for feat in self.feature_list]).T
            self.samples = scaled
        else:
            print("can't set scaled samples, no feature object loaded")

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

    def creat_splits(self, splits_each_feature=None):
        if splits_each_feature is None:
            splits_each_feature = 5
        self.splits = Splits(max_splits_each_feature=splits_each_feature,sample_obj=self)
        for feature in self.feature_list:
            self.splits.create_splits_from_feature(feature)

    def get_splits_obj(self):
        return self.splits

    def get_feature_info(self):
        return [feat.feature_info for feat in self.feature_list]

    def reverse_scale(self, s):
        if not self.feature_list:
            return s
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

    def convert_to_image(self, samples, name='default.png'):
        reverse_scaled = self.reverse_scale(samples)
        labels = reverse_scaled[:, -1]
        value_5s = reverse_scaled[labels == 5.0][:20]
        if value_5s.size == 0:
            return
        decoded_samples = self.decode_samples(value_5s[:, :-1])
        print(f'label of image is {5}')
        for i, sample in enumerate(decoded_samples):
            image = array_to_image(sample)
            if not '.' in name:
                fixed_name = name + f'_{i}.png'
            else:
                fixed_name = name
            save_image_to_folder(image, "output", fixed_name)


#################
#other functions
#################
def value_to_index(array):
    unique_values = np.unique(array)
    indeces = np.zeros(array.shape)
    for i, val in enumerate(unique_values):
        indeces[array == val] = i
    return indeces
