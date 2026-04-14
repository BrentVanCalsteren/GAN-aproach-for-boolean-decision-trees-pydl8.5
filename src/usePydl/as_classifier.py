import numpy as np
from pydl85 import DL85Classifier
import unittest
import helper
from data_obj import dataset


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
        max_depth = 2
        max_bin_len_feat = 4
        num_features = 10
        data = dataset('iris',num_features=num_features,max_bin_len_feat=max_bin_len_feat,y_seperated=True)
        print( f'length of the samples{len(data.x)}')
        classifier = classify_with_default_error(data,max_depth=max_depth,min_sup=1,time=30)
        leafs = helper.get_all_leaves(classifier.tree_)
        print(leafs)
        self.assertLessEqual(len(leafs), 2**max_depth,
        f'number of leaves must be less then{2**max_depth}, is {len(leafs)}')
        self.assertLessEqual(len(leafs), 2**(max_bin_len_feat*num_features),
        f'number of leaves must be less then{2**max_depth}, is {len(leafs)}')

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