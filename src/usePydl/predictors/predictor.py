import numpy as np
from pydl85 import DL85Predictor
import re

from src.usePydl.leaf import get_leafs
from src.usePydl.error_fun import predictor_error, reduce_interval_sizes
from src.usePydl.leaf import just_return_sample_ids
from src.samplers.load_samplers import get_sampler_class

class Predictor:
    def __init__(self,splits,samples, max_depth, min_sup, time):
        self.dl_predictor = DL85Predictor(error_function=reduce_interval_sizes(samples),
                                          leaf_value_function=just_return_sample_ids(),
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)
        self.dl_predictor.fit(splits)

    def predict(self,samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def generate_new_data(self, n_new_samples: int = 100, conf_tresh: float = 0.8,mode: str = "keep_counts") -> np.ndarray:
       #TODO rewrite
       pass

    def create_samplers(self, splits, feature_info):
        tree = self.dl_predictor.tree_
        len_splits = len(splits)
        low_bound = 0
        intervals_each_feature = []
        for i, feat_inf in enumerate(feature_info):
            upp_bound = low_bound + feat_inf[1]
            splits_feat_map = np.zeros((len_splits,2))
            splits_feat_map[low_bound:upp_bound] = np.array([1,i])
            intervals, sample_ids = create_intervals_based_on_tree(tree, splits, splits_feat_map)
            intervals_each_feature.append(intervals)
            low_bound = upp_bound

def create_intervals_based_on_tree(tree,splits,splits_feat_map):
    low_val = 0
    high_val = 1
    intervals_dict_list = []
    traverse_tree(tree, intervals_dict_list,low_val, high_val,splits,splits_feat_map)
    intervals = [item['interval'] for item in intervals_dict_list]
    sample_ids = [item['sample_ids'] for item in intervals_dict_list]

    return intervals, sample_ids


def traverse_tree(node,intervals, low, high,splits,splits_feat_map):
    if 'value' in node:
        sample_ids = node['value'].get('sample_ids', [])
        intervals.append({
            'interval': (low, high),
            'sample_ids': sample_ids
        })
        return

    split_value = node['feat']
    if splits_feat_map[split_value][0] == 1:
        low, high = reduce_interval_of_feat(splits[split_value], low, high)

    if 'left' in node:
        traverse_tree(node['left'], intervals,low, high,splits,splits_feat_map)

    if 'right' in node:
        traverse_tree(node['left'], intervals,low, high,splits,splits_feat_map)

def reduce_interval_of_feat(value_str, low, high):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", value_str)
    vals = [float(x) for x in matches]
    if 'bigger' in value_str:
        low = vals[0]
    elif 'smaller' in value_str:
        high = vals[0]
    elif 'even' in value_str:
        low = vals[0]
        high = vals[0]
    elif 'interval' in value_str:
        low = vals[0]
        high = vals[1]
    else:
        print('invalid key')
    return low, high
"""
        features = np.array(samples[list(tids)]).T
        feature_type_dic = {}
        for i, feat_type in enumerate(sampler_types):
            if feat_type not in feature_type_dic:
                feature_type_dic[feat_type] = [i]
            else:
                feature_type_dic[feat_type].append(i)

        samplers_list = []
        for feat_type, indices in feature_type_dic.items():
            sample_class = get_sampler_class(feat_type)
            if sample_class:
                all_feat_same_type = features[indices]
                fitted_samplers = sample_class.fit_all_features_of_this_type(all_feat_same_type)
                samplers_list.append({
                    "feat_ids": indices,
                    "samplers": fitted_samplers,
                    "num_feat": features.shape[0],
                    "sample_class": sample_class,
                })

"""