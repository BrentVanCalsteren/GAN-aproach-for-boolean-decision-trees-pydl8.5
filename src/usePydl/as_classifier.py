import numpy as np
from pydl85 import DL85Classifier
import src.dataLoader.dataset_loader as loader
import src.binaryConvertion.binner as binner
import unittest
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


def classify_with_default_error(data,max_depth,min_sup,time):
    clasfi = DL85Classifier(max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(data.x_bin, data.y)
    return clasfi


def classify_with_custom_error(data,error_fun,max_depth,min_sup,time):
    clasfi = DL85Classifier(fast_error_function=error_fun,max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(data.x_bin, data.y)
    return clasfi


####################################
##### error functions #############
###################################
def sim_error(y):
    def error(tids):
        supports = list(tids)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex], maxindex
    return error

##########################
####### tests ############
#########################
class TestClassify(unittest.TestCase):
    def test_classify_standard(self):
        max_depth = 4
        max_bin_len_feat = 2
        num_features = 1
        data = dataset('iris',num_features=num_features,max_bin_len_feat=max_bin_len_feat,y_seperated=True)
        classifier = classify_with_default_error(data,max_depth=1,min_sup=1,time=30)
        leafs = helper.get_all_leaves(classifier.tree_)
        self.assertLessEqual(len(leafs), 2**max_depth,
        f'number of leaves just be less then{2**max_depth}, is {len(leafs)}')
        self.assertLessEqual(len(leafs), 2**(max_bin_len_feat*num_features),
        f'number of leaves just be less then{2**max_depth}, is {len(leafs)}')

    def test_classify_simulate_classify_error(self):
        max_depth = 4
        max_bin_len_feat = 2
        num_features = 1
        data = dataset('iris',num_features=num_features,max_bin_len_feat=max_bin_len_feat,y_seperated=True)
        error = sim_error(data.y)
        classifier = classify_with_custom_error(data,error,max_depth,min_sup=1,time=30)
        leafs = helper.get_all_leaves(classifier.tree_)
        self.assertLessEqual(len(leafs), 2**max_depth,
        f'number of leaves just be less then{2**max_depth}, is {len(leafs)}')
        self.assertLessEqual(len(leafs), 2**(max_bin_len_feat*num_features),
        f'number of leaves just be less then{2**max_depth}, is {len(leafs)}')

if __name__ == "__main__":
    unittest.main()