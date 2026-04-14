import src.dataLoader.dataset_loader as loader
import src.binaryConvertion.binner as binner
from sklearn.model_selection import train_test_split
import numpy as np
import random

class dataset:
    def __init__(self,dataset_name,num_features=10,max_bin_len_feat=10,
                 y_seperated=True,y_index= -1):
        dataset = loader.load_dataloader_by_name(
            dataset_name, y_seperated=y_seperated, y_index=y_index)
        x_complete = dataset.get_x_complete()
        x_missing = dataset.get_x_missing()
        y_complete = dataset.get_y_complete()
        print(f"y normaly: {y_complete}")
        self.y = loader.standardize_to_num(y_complete).astype(np.int32)
        print(f"y standardized: {self.y}")
        y_missing = dataset.get_y_missing()
        x_scaled_T = loader.standardize_2d_array(x_complete.T)
        x_scaled_T_clamped = x_scaled_T[:num_features,:] #schrinking the number of features to work with
        bin_string_x, bin_length_x, clusters = binner.bin_convertion_2d(x_scaled_T_clamped, max_bins=max_bin_len_feat)
        self.x_bin = np.array([binner.flatten_binary_strings(row) for row in bin_string_x.T])
        x_scaled = x_scaled_T_clamped.T
        self.x = x_scaled

def randomize_data(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random.randint(1, 100))
    return X_train, X_test, y_train, y_test