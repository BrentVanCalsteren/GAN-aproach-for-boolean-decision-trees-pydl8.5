from typing import List

import numpy as np
from pydl85 import DL85Predictor
import re

import itertools
from src.usePydl.leaf import get_leafs
from src.usePydl.error_fun import predictor_error, reduce_interval_sizes, mse_error
from src.usePydl.leaf import just_return_sample_ids
from src.samplers.load_samplers import get_sampler_class
from src.samplers.multinomial import MultinomialSampler
from src.samplers.uniform import UniformSampler
from src.samplers.single_gaussian import SingleGaussian1DSampler
from src.samplers.multivariate_gaussian import MultivariateGaussianSampler
from typing import List, Dict, Any

MULTI = False
COMBINE_FEAT = False

class Predictor:
    n_samples = None

    def __init__(self, splits, samples, max_depth, min_sup, time, n_samples=None):
        print('starting dl predictor...')
        if n_samples:
            self.n_samples = n_samples
        else:
            self.n_samples = samples.shape[0]
        self.dl_predictor = DL85Predictor(error_function=reduce_interval_sizes(samples),
                                          leaf_value_function=just_return_sample_ids(self.n_samples),
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)
        self.dl_predictor.fit(splits)

    def predict(self, samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def get_dl_tree(self):
        return self.dl_predictor.tree_

    def gen_new_data(self, splits=None, feature_info=None, n_new_samples: int = 100,
                     conf_tresh: float = 0.8) -> np.ndarray:
        return self.gen_new_data_based_on_tree(splits=splits,
                                               feature_info=feature_info,
                                               n_new_samples=n_new_samples,
                                               conf_tresh=conf_tresh)

    def gen_new_data_based_on_tree(self, tree=None, splits=None, feature_info=None, n_new_samples: int = 100,
                                   conf_tresh: float = 0.8, old_samples = None) -> np.ndarray:
        if not tree:
            tree = self.get_dl_tree()
        len_splits = len(splits)
        splits_feat_map = np.zeros(len_splits)
        low_bound = 0
        for i, feat_inf in enumerate(feature_info):
            upp_bound = low_bound + feat_inf[1]
            splits_feat_map[low_bound:upp_bound] = np.array([i] * feat_inf[1])
            low_bound = upp_bound
        paths = get_all_paths(tree)
        path_sample_dic = {}
        for i, path in enumerate(paths):
            path_sample_dic[i] = [path, create_intervals_of_path(path, splits, splits_feat_map, len(feature_info))]
        return gen_new_data(path_sample_dic, feature_info, n=n_new_samples, conf=conf_tresh, samples=old_samples)


def create_intervals_of_path(path: Dict[str, Any], splits: List[str],
                             splits_feat_map, num_features: int):
    intervals_each_feature = [[[0.0, 1.0]] for _ in range(num_features)]
    for direction, split_idx in path['path_steps']:
        feature_id = int(splits_feat_map[split_idx])
        split_string = splits[split_idx]

        if direction == 'L':
            intervals_each_feature[feature_id] = update_left_splits(
                split_string, intervals_each_feature[feature_id]
            )
        elif direction == 'R':
            intervals_each_feature[feature_id] = update_right_splits(
                split_string, intervals_each_feature[feature_id]
            )

    return intervals_each_feature


def inbetween(interval: List[float], val: float) -> bool:
    return interval[0] <= val <= interval[1]


def parse_values(value_str: str) -> List[float]:
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", value_str)
    return [float(x) for x in matches]


def update_left_splits(value_str: str, intervals: List[List[float]]) -> List[List[float]]:
    vals = parse_values(value_str)
    if not vals:
        return intervals
    new_intervals = []
    for inter in intervals:
        a, b = inter[0], inter[1]
        if 'bigger' in value_str:
            new_a = max(a, vals[0])
            if new_a <= b:
                new_intervals.append([new_a, b])
        elif 'smaller' in value_str:
            new_b = min(b, vals[0])
            if a <= new_b:
                new_intervals.append([a, new_b])
        elif 'even' in value_str:
            new_a = max(a, vals[0])
            new_b = min(b, vals[0])
            if new_a <= new_b:
                new_intervals.append([new_a, new_b])
        elif 'interval' in value_str and len(vals) >= 2:
            new_a = max(a, vals[0])
            new_b = min(b, vals[1])
            if new_a <= new_b:
                new_intervals.append([new_a, new_b])
        else:
            print('Invalid key or missing values')
            new_intervals.append(inter)
    return new_intervals


def update_right_splits(value_str: str, intervals: List[List[float]]) -> List[List[float]]:
    delta = 0
    vals = parse_values(value_str)
    if not vals:
        return intervals
    new_intervals = []
    for inter in intervals:
        a, b = inter[0], inter[1]
        if 'bigger' in value_str:
            new_b = min(b, vals[0] - delta)
            if a <= new_b:
                new_intervals.append([a, new_b])
        elif 'smaller' in value_str:
            new_a = max(a, vals[0] + delta)
            if new_a <= b:
                new_intervals.append([new_a, b])
        elif 'even' in value_str:
            new_b1 = min(b, vals[0] - delta)
            if a <= new_b1:
                new_intervals.append([a, new_b1])
            new_a2 = max(a, vals[0] + delta)
            if new_a2 <= b:
                new_intervals.append([new_a2, b])
        elif 'interval' in value_str and len(vals) >= 2:
            new_b1 = min(b, vals[0] - delta)
            if a <= new_b1:
                new_intervals.append([a, new_b1])
            new_a2 = max(a, vals[1] + delta)
            if new_a2 <= b:
                new_intervals.append([new_a2, b])
        else:
            print('Invalid key or missing values')
            new_intervals.append(inter)
    return new_intervals


def get_all_paths(tree):
    all_paths = []

    def path_finder(node, current_path):
        if 'value' in node:
            all_paths.append({
                'path_steps': list(current_path),
                'error': node.get('error', 0),
                'sample_ids': node['value']['sample_ids'],
                'rel_prob': node['value']['rel_prob']
            })
            return

        feature = node['feat']
        if 'left' in node:
            current_path.append(["L", feature])
            path_finder(node['left'], current_path)
            current_path.pop()
        if 'right' in node:
            current_path.append(["R", feature])
            path_finder(node['right'], current_path)
            current_path.pop()

    path_finder(tree, [])
    return all_paths


def gen_new_data(path_samples_dict, feature_info, n, conf, samples=None):
    probs_each_path = np.array([intervals[0]['rel_prob'] for pathid, intervals in path_samples_dict.items()])
    prob_sum = np.sum(probs_each_path)
    if prob_sum > 0:
        probs_each_path /= prob_sum
    else:
        probs_each_path = np.ones(len(probs_each_path)) / len(probs_each_path)

    intervals_each_path = [intervals[1] for pathid, intervals in path_samples_dict.items()]
    disc_feat_ids = [feat_inf[0] is not None for feat_inf in feature_info]
    cont_feat_ids = [feat_inf[0] is None for feat_inf in feature_info]
    samples_disc = samples[:,disc_feat_ids]
    samples_cont = samples[:,cont_feat_ids]

    all_new_samples = np.array([])
    path_indices = np.random.choice(probs_each_path.size, n, p=probs_each_path)
    _, counts = np.unique(path_indices, return_counts=True)
    for i, count in enumerate(counts):
        gen_feat_matrix = np.zeros((len(feature_info), count))
        intervals_each_feature = intervals_each_path[i]
        path = path_samples_dict[i][0]
        sample_ids = path.get('sample_ids', [])
        disc_feats = samples_disc[sample_ids].T
        cont_feats = samples_cont[sample_ids].T
        #intervals_disc = intervals_each_feature[disc_feat_ids]
        #intervals_cont = intervals_each_feature[cont_feat_ids]

        disc_samplers = []
        for i in range(disc_feats.shape[0]):
            sampler = MultinomialSampler()
            sampler.fit(disc_feats[i])
            disc_samplers.append(sampler)
        MultinomialSampler.generate_new_samples_for_all_features_of_this_type(
                                                indices=disc_feat_ids,
                                                gen_feats_matrix=gen_feat_matrix,
                                                conf_thresh=conf,
                                                samplers=disc_samplers,)

        cont_samplers = []
        if COMBINE_FEAT:
            sampler = MultivariateGaussianSampler()
            sampler.fit(cont_feats.T)
            cont_samplers.append(sampler)
            MultivariateGaussianSampler.generate_new_samples_for_all_features_of_this_type(
                                                    indices=cont_feat_ids,
                                                    gen_feats_matrix=gen_feat_matrix,
                                                    conf_thresh=conf,
                                                    samplers=cont_samplers,)
        else:
            for i in range(cont_feats.shape[0]):
                sampler = SingleGaussian1DSampler()
                sampler.fit(cont_feats[i])
                cont_samplers.append(sampler)
            SingleGaussian1DSampler.generate_new_samples_for_all_features_of_this_type(
                indices=cont_feat_ids,
                gen_feats_matrix=gen_feat_matrix,
                conf_thresh=conf,
                samplers=cont_samplers, )

        if all_new_samples.size > 0:
            all_new_samples = np.vstack((all_new_samples,gen_feat_matrix.T))
        else: all_new_samples = gen_feat_matrix.T
    return np.clip(all_new_samples, 0, 1)
