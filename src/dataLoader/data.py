import numpy as np

import src.binaryConvertion.binner as binner
from src.usePydl.predictor.gaussian_predictor import GaussianPredictor
from sklearn.model_selection import train_test_split
import random


class Data:
    x = None
    x_bin = None
    x_clusters = None

    x_gen = None
    x_gen_bin = None

    split_feature = []
    split_data_array = []

    predictor = None

    def __init__(self, feat_scaled,bin_length=-1):
        if bin_length == -1:
            bin_length = calculate_bin_length(feat_scaled)
            print(f'bin_length is not given, choosing own lenght:{bin_length}')
        self.x = feat_scaled.T
        self.x_bin, self.x_clusters = binner.bin_convertion_2d(feat_scaled,max_bins=bin_length)

    def get_data_at_depth(self,depth=-1):
        if depth == -1:
            return {"last":self}
        x_array = []
        for i,data in enumerate(self.split_data_array):
            data = data.get_x_at_depth(depth-1)
            x_array.append({self.split_feature[i]:data})
        return x_array

    def split_data_on_index(self,index=None):
        features = self.x.T
        if index == None:
            print('no index given, finding best split')
            index = get_min_feat_index(features)
        self.split_feature = features[index, :]
        reduced = np.delete(features, index, axis=0)

        unique_vals = np.unique(self.split_feature)
        result = []
        for val in unique_vals:
            mask = (self.split_feature == val)
            result.append(reduced[:, mask])
        for sub_features in result:
            self.split_data_array.append(Data(sub_features))
        print(f'data succesfully split on feature: {index},new data_objs: {self.split_data_array}')

    def shuffle_data(self):
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.x_bin = self.x_bin[p]

    def load_predictor(self,predictor_name,max_depth=3,min_sup=1,time=100):
        if predictor_name == "gaussian":
            self.predictor = GaussianPredictor(self.x,self.x_bin,max_depth=max_depth,min_sup=min_sup,time=time)

    def generate_more_data(self,n=100,conf=0.8):
       if self.predictor is None:
           raise Exception("predictor cannot be None")
       else: self.predictor.generate_new_data(n_new_samples=n,conf_tresh=conf)

 #HELPERs
def get_min_feat_index(features):
    min_indx = 0
    min_uniques = np.inf
    for i, feature in enumerate(features):
        uniques = np.unique(feature)
        if len(uniques) <= min_uniques:
            min_indx = i
            min_uniques = len(uniques)
    return min_indx

def calculate_bin_length(features_scaled):
    min_uniques = np.inf
    for feature in features_scaled:
        uniques = np.unique(feature)
        if len(uniques) <= min_uniques:
            min_uniques = len(uniques)
    return min_uniques * 2

def convert_feats_specific_bin_length(feats, clusters):
    bin_data = []
    for i in range(len(feats)):
        bin_data.append(binner.gen_one_hot_string(feats[i], clusters[i]))
    #print(bin_data)
    return np.array([binner.flatten_binary_strings(row) for row in np.array(bin_data).T])

def split_train_test(X, Y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                        random_state=random.randint(1, 100))
    return X_train, X_test, y_train, y_test

