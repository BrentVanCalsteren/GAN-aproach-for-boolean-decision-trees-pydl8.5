import src.dataLoader.dataset_loader as loader
import src.binaryConvertion.binner as binner
from sklearn.model_selection import train_test_split
import numpy as np
import random

class dataset:
    x_gen = None
    x_gen_bin = None
    y_gen_scaled = None
    y_gen = None


    def __init__(self,**kwargs):
        self.x = None
        self.x_bin = None
        self.x_clusters = None
        self.y = None
        self.y_scaled = None
        self.reload_data(**kwargs)
        self.is_shifted = True


    def reload_data(self,dataset_name='iris',num_features=10,max_bin_len_feat=10,
                 y_seperated=True,y_index= -1,shifted=True):
        loaded_data = loader.load_dataloader_by_name(
            dataset_name, y_seperated=y_seperated, y_index=y_index)
        x_complete = loaded_data.get_x_complete()
        x_missing = loaded_data.get_x_missing()
        y_complete = loaded_data.get_y_complete()
        print(f"y normaly: {y_complete}")
        self.y_scaled = loader.standardize_to_num(y_complete)
        if self.y_scaled is not None:
            self.y = loader.value_to_index_array(self.y_scaled)
        print(f"y standardized: {self.y}")
        y_missing = loaded_data.get_y_missing()
        x_scaled_T = loader.standardize_2d_array(x_complete.T)
        x_scaled_T_clamped = x_scaled_T[:num_features, :]  # schrinking the number of features to work with
        bin_string_x, bin_length_x, self.x_clusters = binner.bin_convertion_2d(x_scaled_T_clamped,
                                                                               max_bins=max_bin_len_feat)
        self.x_bin = np.array([binner.flatten_binary_strings(row) for row in bin_string_x.T])
        x_scaled = x_scaled_T_clamped.T
        self.x = x_scaled


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

    def shuffle_data(self):
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.x_bin = self.x_bin[p]
        if self.y is not None:
            self.y = self.y[p]
            self.y_scaled = self.y_scaled[p]


def convert_feats_specific_bin_length(feats, clusters):
    bin_data = []
    for i in range(len(feats)):
        bin_data.append(binner.gen_one_hot_string(feats[i], clusters[i]))
    print(bin_data)
    return np.array([binner.flatten_binary_strings(row) for row in np.array(bin_data).T])

def split_train_test(X, Y,test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random.randint(1, 100))
    return X_train, X_test, y_train, y_test