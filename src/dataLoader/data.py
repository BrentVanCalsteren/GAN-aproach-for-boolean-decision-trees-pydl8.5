from typing import List

import numpy as np
import src.binaryConvertion.binner as binner
from src.usePydl.predictor.gaussian_multi_predictor import GaussianMultiPredictor
from src.usePydl.predictor.uniform_predictor import UNiPredictor
from src.usePydl.predictor.gaussian_1D_predictor import Gaussian1DPredictor
from src.usePydl.predictor.mixed_predictor import MixedPredictor
from src.usePydl.predictor.ensemble_predictors import EnsemblePredictor
from src.dataLoader.feature_struct import FeatureStruct

MIN_SPLIT_NUMBER = 20

class Data:
    x = None
    x_bin = None
    feature_bin_len = None
    feature_clusters = None

    x_gen = None
    x_gen_bin = None

    split_feature : FeatureStruct = None #is the feature that this subdata is split with
    parent_data = None
    depth = 0

    predictor = None
    discrete_feature_ids = None

    def __init__(self, feat_scaled,bin_length=-1,split_feature=None,parent_data=None,depth=0):
        self.depth = depth
        self.child_datas = []
        self.x = feat_scaled.T
        self.DISCRETE_LIMIT = self.x.shape[0]//20
        self.discrete_feature_ids = self.check_discrete_features(feat_scaled)
        self.split_feature = split_feature
        self.parent_data = parent_data
        if bin_length == -1:
            bin_length = calculate_bin_length(feat_scaled)
            print(f'bin_length is not given, choosing own lenght:{bin_length}')
        self.x_bin, self.feature_bin_len,self.feature_clusters = binner.bin_convertion_2d(feat_scaled, max_bins=bin_length)
        self.shuffle_samples()

    def get_data_at_depth(self, data_array, target_depth):
        if self.depth == target_depth:
            data_array.append(self)
        elif self.depth < target_depth:
            if self.child_datas:
                for child in self.child_datas:
                    child.get_data_at_depth(data_array, target_depth)
            else:
                data_array.append(self)

    def split_data_on_index(self,index=None):
        features = self.x.T
        if index == None:
            print('no index given, finding best split')
            index = get_min_feat_index(features)
            print(f'INDEX FOUND: {index}')
        if index not in self.discrete_feature_ids:
            print(f"Can't split data on continue val")
            return
        if self.x.shape[0] < MIN_SPLIT_NUMBER:
            print(f"Can't split data further, too few samples")
            return
        split_features = features[index, :]
        reduced = np.delete(features, index, axis=0)

        unique_vals = np.unique(split_features)
        result = []
        for val in unique_vals:
            mask = (split_features == val)
            result.append(reduced[:, mask])
        for i, sub_features in enumerate(result):
            self.child_datas.append(
                Data(
                    depth=self.depth+1,
                    feat_scaled=sub_features,
                    split_feature=FeatureStruct(val=float(unique_vals[i]),feat_index=int(index)),
                    parent_data=self))
        print(f'data succesfully split on feature: {index},new n data_objs: {len(self.child_datas)}')

    def shuffle_samples(self):
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.x_bin = self.x_bin[p]

    def load_predictor(self, predictor_type:str, max_depth=3, min_sup=1, time=100):
        print('loading predictor...')
        if predictor_type == "gaussian_multi":
            self.predictor = GaussianMultiPredictor(self.x,self.x_bin,max_depth=max_depth,min_sup=min_sup,time=time)
        elif predictor_type == "uniform":
            self.predictor = UNiPredictor(self.x,self.x_bin,max_depth=max_depth,min_sup=min_sup,time=time)
        elif predictor_type == "gaussian_1D":
            self.predictor = Gaussian1DPredictor(self.x, self.x_bin, max_depth=max_depth, min_sup=min_sup, time=time)
        elif predictor_type == "mixed":
            self.predictor = MixedPredictor(self.x, self.x_bin,discrete_feature_ids=self.discrete_feature_ids, max_depth=max_depth, min_sup=min_sup, time=time)
        elif predictor_type == "ensemble":
            self.predictor = EnsemblePredictor(self.x,self.x_bin)
        else:
            raise Exception('predictor type not found')

    def generate_more_data(self,n=200,conf=0.8):
       print("Generating data...")
       if self.predictor is None:
           raise Exception("predictor cannot be None")
       else:
            x_gen = self.predictor.generate_new_data(n_new_samples=n,conf_tresh=conf)
            self.x_gen = self.make_discrete(x_gen)
            self.set_bin_x_gen()

    def make_discrete(self, x_gen):
        features_gen = x_gen.T
        features_real = self.x.T
        converted_features = []
        for i,feature in enumerate(features_gen):
            if i in self.discrete_feature_ids:
                print(f"discritisising gen feature: {i}")
                uniques = np.unique(features_real[i])
                conv_feature = []
                for f in feature:
                    dists = np.abs(uniques - f)
                    conv_feature.append(uniques[np.argmin(dists)])
                converted_features.append(conv_feature)
            else: converted_features.append(feature)
        return np.array(converted_features).T


    def set_bin_x_gen(self):
        if self.x_gen is None:
            return
        features = self.x_gen.T
        one_hots = []
        for i, feature in enumerate(features):
            one_hots.append(binner.gen_one_hot_string(feature, self.feature_clusters[i]))
        self.x_gen_bin = np.array([binner.flatten_binary_strings(row) for row in np.array(one_hots).T])

    def check_discrete_features(self, feat_scaled):
        descrete_ids = []
        for i, feature in enumerate(feat_scaled):
            uniques = np.unique(feature)
            if len(uniques) <= self.DISCRETE_LIMIT:
                descrete_ids.append(i)
        return descrete_ids


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


