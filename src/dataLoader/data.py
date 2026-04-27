import numpy as np
import src.binaryConvertion.binner as binner
from src.usePydl.predictor.gaussian_multi_predictor import GaussianMultiPredictor
from src.usePydl.predictor.uniform_predictor import UNiPredictor
from src.usePydl.predictor.gaussian_1D_predictor import Gaussian1DPredictor
from src.dataLoader.feature_struct import FeatureStruct

DISCRETE_LIMIT = 5

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
        self.discrete_feature_ids = check_discrete_features(feat_scaled)
        self.split_feature = split_feature
        self.parent_data = parent_data
        if bin_length == -1:
            bin_length = calculate_bin_length(feat_scaled)
            print(f'bin_length is not given, choosing own lenght:{bin_length}')
        self.x = feat_scaled.T
        self.x_bin, self.feature_bin_len,self.feature_clusters = binner.bin_convertion_2d(feat_scaled, max_bins=bin_length)
        self.shuffle_samples()

    def get_data_at_depth(self, data_array, target_depth):
        if self.depth == target_depth:
            data_array.append(self)
        elif self.depth < target_depth:
            for child in self.child_datas:
                child.get_data_at_depth(data_array, target_depth)

    def split_data_on_index(self,index=None):
        features = self.x.T
        if index == None:
            print('no index given, finding best split')
            index = get_min_feat_index(features)
            print(f'INDEX FOUND: {index}')
        if index not in self.discrete_feature_ids:
            print(f"Can't split data on continue val")
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

    def load_predictor(self,predictor_name,max_depth=3,min_sup=1,time=100):
        print('loading predictor...')
        if predictor_name == "gaussian_multi":
            self.predictor = GaussianMultiPredictor(self.x,self.x_bin,max_depth=max_depth,min_sup=min_sup,time=time)
        elif predictor_name == "uniform":
            self.predictor = UNiPredictor(self.x,self.x_bin,max_depth=max_depth,min_sup=min_sup,time=time)
        elif predictor_name == "gaussian_1D":
            self.predictor = Gaussian1DPredictor(self.x, self.x_bin, max_depth=max_depth, min_sup=min_sup, time=time)
        else:
            raise Exception('predictor type not found')

    def generate_more_data(self,n=200,conf=0.8):
       print("Generating data...")
       if self.predictor is None:
           raise Exception("predictor cannot be None")
       else:
            self.x_gen = self.predictor.generate_new_data(n_new_samples=n,conf_tresh=conf)
            self.set_bin_x_gen()


    def set_bin_x_gen(self):
        if self.x_gen is None:
            return
        features = self.x_gen.T
        one_hots = []
        for i, feature in enumerate(features):
            one_hots.append(binner.gen_one_hot_string(feature, self.feature_clusters[i]))
        self.x_gen_bin = np.array([binner.flatten_binary_strings(row) for row in np.array(one_hots).T])


 #HELPERs

def check_discrete_features(feat_scaled):
    descrete_ids = []
    for i, feature in enumerate(feat_scaled):
        uniques = np.unique(feature)
        if len(uniques) <= DISCRETE_LIMIT:
            descrete_ids.append(i)
    return descrete_ids

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


