import src.dataLoader.dataset_loader as loader
from sklearn.model_selection import train_test_split
import numpy as np
import random
from data import Data

class dataset:
    samples_complete = None
    samples_missing = None
    data = None


    def __init__(self,**kwargs):
        self.reload_data(**kwargs)


    def reload_data(self,dataset_name='iris'):
        loaded_data = loader.load_dataloader_by_name(dataset_name)
        self.samples_complete = loaded_data.get_x_complete()
        self.samples_missing = loaded_data.get_x_missing()
        features_scaled = loader.standardize_2d_features(self.samples_complete.T)
        self.data = Data(features_scaled)


    def add_gen_data(self,gen_data,y_index=-1):
        #x_gen
        y_gen = gen_data[:, y_index]
        self.x_gen = np.delete(gen_data, y_index, axis=1)
        self.x_gen_bin = convert_feats_specific_bin_length(self.x_gen.T, self.x_clusters)
        #y_gen
        if self.y is not None:
            self._map_y_to_closest_val(y_gen)

    def _map_y_to_closest_val(self,y_gen):
        unique_values = np.unique(self.y_scaled)
        y_mapped = np.zeros(y_gen.shape)
        for i,y in enumerate(y_gen):
            y_mapped[i] = np.argsort(np.abs(unique_values-y))[0]
        self.y_gen_scaled = y_mapped
        self.y_gen = loader.value_to_index_array(y_mapped)







