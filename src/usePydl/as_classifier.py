import numpy as np
from pydl85 import DL85Classifier
import src.dataLoader.dataset_loader as loader
import src.binaryConvertion.binner as binner
import helper
from sklearn.metrics import accuracy_score, classification_report

class dataset:
    def __init__(self,dataset_name,num_features=10,max_bin_len_feat=10,
                 y_seperated=True,y_index= -1):
        dataset = loader.load_dataloader_by_name(
            dataset_name, y_seperated=y_seperated, y_index=y_index)
        x_complete = dataset.get_x_complete()
        x_missing = dataset.get_x_missing()
        y_complete = dataset.get_y_complete()
        self.y = loader.standardize_to_num(y_complete)
        y_missing = dataset.get_y_missing()
        x_scaled_T = loader.standardize_2d_array(x_complete.T)
        x_scaled_T_clamped = x_scaled_T[:num_features,:] #schrinking the number of features to work with
        bin_string_x, bin_length_x, clusters = binner.bin_convertion_2d(x_scaled_T_clamped, max_bins=max_bin_len_feat)
        self.x_bin = np.array([binner.flatten_binary_strings(row) for row in bin_string_x.T])
        x_scaled = x_scaled_T_clamped.T
        self.x = x_scaled


def classify_with_default_error(dataset,max_depth,min_sup,time):
    clasfi = DL85Classifier(max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(dataset.x_bin, dataset.y)
    return clasfi


def classify_with_custom_error(dataset,error_fun,max_depth,min_sup,time):
    clasfi = DL85Classifier(error_function=error_fun,max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(dataset.x_bin, dataset.y)
    return clasfi


####################################
##### error functions #############
###################################

def simulate_classify_error(y):
    def error(tids):
        classes, supports = np.unique(y.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex]
    return error
