import re
from typing import Dict, List

import numpy as np
from scipy._lib.pyprima.cobyla import initialize

from src.data.splits import Splits



class Tree:

    def __init__(self, tree, split_obj):
        self.tree : Dict = tree
        self.split_obj : Splits= split_obj

    def extend_tree(self, subtree, leaf_id):
        current_leaf_id = 0
        def merge(node):
            nonlocal current_leaf_id
            if "value" in node:
                if leaf_id == current_leaf_id:
                    return subtree
                current_leaf_id += 1
                return node

            if "left" in node:
                node["left"] = merge(node["left"])
            if "right" in node:
                node["right"] = merge(node["right"])
            return node
        return merge(self.tree)

    def get_leafs(self):
        leaves = []
        def recurse(node):
            if "value" in node:
                leaves.append(node)
            else:
                recurse(node["left"])
                recurse(node["right"])
        recurse(self.tree)
        return leaves

    def get_intervals_each_path(self):
        paths = self.get_all_paths()
        interval_path_dic = {}
        for i, path in enumerate(paths):
            interval_path_dic[i] = self.calc_intervals_of_path(path['path'])
        return interval_path_dic

    def get_all_paths(self):
        all_paths = []
        def path_finder(node, current_path):
            if 'value' in node:
                all_paths.append(('END',{
                    'path': current_path, # is a dic feat_id : (splits, directions)
                    'error': node.get('error', 0),
                    'sample_ids': node['value']['sample_ids'],
                    'rel_prob': node['value']['rel_prob']
                }))
                return

            split = int(node['feat'])
            feature_id = self.split_obj.feature_index_array[split]
            if current_path[feature_id] is not None:
                splits, directions = current_path[feature_id]
            else:
                splits, directions = [], []
            if 'left' in node:
                splits.append(split)
                directions.append('L')
                current_path[feature_id] = (splits,directions)
                path_finder(node['left'], current_path)
                splits.pop()
                directions.pop()
            if 'right' in node:
                splits.append(split)
                directions.append('R')
                current_path[feature_id] = (splits,directions)
                path_finder(node['right'], current_path)
                splits.pop()
                directions.pop()

        path_finder(self.tree, {})
        if len(all_paths) != len(self.get_leafs()):
            print('LEN PATHS AND TREES SHOULD BE TE SAME')
        return all_paths

    def calc_intervals_of_path(self, path):
        feat_interval_dic = {}
        for feature_id in range(np.max(self.split_obj.feature_index_array)):
            start_interval =  [[0.0,1.0]]
            feat_interval_dic[feature_id] = expand_interval(path[feature_id], start_interval)
        return feat_interval_dic

def expand_interval(path, start_interval):
    intervals = start_interval
    for i in range(len(path[0])):
        if path[1][i] == 'R':
            intervals = update_right_splits(path[1][i], intervals)
        else:
            intervals = update_left_splits(path[1][i], intervals)
    return intervals


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
        if 'bigger_eq' in value_str:
            new_a = max(a, vals[0])
            if new_a <= b:
                new_intervals.append([new_a, b])
        elif 'smaller_eq' in value_str:
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
    delta = 1e-6
    vals = parse_values(value_str)
    if not vals:
        return intervals
    new_intervals = []
    for inter in intervals:
        a, b = inter[0], inter[1]
        if 'bigger_eq' in value_str:
            new_b = min(b, vals[0] - delta)
            if a <= new_b:
                new_intervals.append([a, new_b])
        elif 'smaller_eq' in value_str:
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