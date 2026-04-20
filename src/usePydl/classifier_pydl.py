from pydl85 import DL85Classifier
import unittest
import leaf
from src.dataLoader.Dataset import dataset
from error_fun import min_sup_error


######################"
#clasifiers
##################

def classify_with_default_error(x_bin,y,max_depth=3,min_sup=2,time=100):
    clasfi = DL85Classifier(max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(x_bin, y)
    return clasfi


def classify_with_custom_error(x_bin,y,error_fun,max_depth,min_sup,time):
    clasfi = DL85Classifier(fast_error_function=error_fun,max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(x_bin, y)
    return clasfi



##########################
####### tests ############
#########################
class TestClassify(unittest.TestCase):
    def test_classify_standard(self):
        max_depth = 2
        max_bin_len_feat = 4
        num_features = 10
        data = dataset(dataset_name='iris',num_features=num_features,max_bin_len_feat=max_bin_len_feat,y_seperated=True)
        print( f'length of the samples{len(data.x)}')
        classifier = classify_with_default_error(data.x_bin,data.y,max_depth=max_depth,min_sup=1,time=30)
        leafs = leaf.get_all_leaves(classifier.tree_)
        print(leafs)
        self.assertLessEqual(len(leafs), 2**max_depth,
        f'number of leaves must be less then{2**max_depth}, is {len(leafs)}')
        self.assertLessEqual(len(leafs), 2**(max_bin_len_feat*num_features),
        f'number of leaves must be less then{2**max_depth}, is {len(leafs)}')

    def test_classify_simulate_classify_error(self):
        max_depth = 4
        max_bin_len_feat = 2
        num_features = 1
        data = dataset(dataset_name='iris',num_features=num_features,max_bin_len_feat=max_bin_len_feat,y_seperated=True)
        error = min_sup_error(data.y)
        classifier = classify_with_custom_error(data.x_bin,data.y,error,max_depth,min_sup=1,time=30)
        leafs = leaf.get_all_leaves(classifier.tree_)
        self.assertLessEqual(len(leafs), 2**max_depth,
        f'number of leaves just be less then{2**max_depth}, is {len(leafs)}')
        self.assertLessEqual(len(leafs), 2**(max_bin_len_feat*num_features),
        f'number of leaves just be less then{2**max_depth}, is {len(leafs)}')

if __name__ == "__main__":
    unittest.main()