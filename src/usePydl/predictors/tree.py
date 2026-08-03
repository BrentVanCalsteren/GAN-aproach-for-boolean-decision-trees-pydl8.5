import re
from typing import Dict, List

import numpy as np

from src.usePydl.predictors.interval import Intervals


class Tree:

    def __init__(self, tree):
        self.tree : Dict = tree


    def get_leaf_id_for_sample(self, sample):
        if sample is None or self.tree is None:
            return None

        sample_arr = np.asarray(sample)
        if sample_arr.ndim == 2:
            return np.array([self.get_leaf_id_for_sample(s) for s in sample_arr])

        def traverse(node):
            if "value" in node:
                return node["value"].get("leaf_id", 0)

            feat_id = int(node['feat'])
            split_val = node.get('split_val', '')
            val = sample_arr[feat_id]

            if isinstance(split_val, str):
                vals = parse_values(split_val)
                if not vals:
                    cond = True
                elif 'bigger_eq' in split_val:
                    cond = (val >= vals[0])
                elif 'smaller_eq' in split_val:
                    cond = (val <= vals[0])
                elif 'even' in split_val:
                    cond = (val == vals[0])
                elif 'interval' in split_val and len(vals) >= 2:
                    cond = (vals[0] <= val <= vals[1])
                else:
                    cond = (val <= vals[0])
            else:
                cond = (val <= float(split_val))

            if cond:
                return traverse(node["left"]) if "left" in node else traverse(node["right"])
            else:
                return traverse(node["right"]) if "right" in node else traverse(node["left"])

        return traverse(self.tree)

    def check_tree_purity_with_samples(self, samples, labels):
        if samples is None or labels is None or len(samples) == 0:
            return 0.0, {}

        labels_arr = np.asarray(labels).flatten()
        samples_arr = np.asarray(samples)
        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(1, -1)

        leaf_ids = self.get_leaf_id_for_sample(samples_arr)

        leaf_label_map = {}
        for l_id, label in zip(leaf_ids, labels_arr):
            if l_id not in leaf_label_map:
                leaf_label_map[l_id] = []
            leaf_label_map[l_id].append(label)

        leaf_purity_info = {}
        total_purity_sum = 0.0
        n_leaves = len(leaf_label_map)

        for l_id, l_labels in leaf_label_map.items():
            l_labels = np.array(l_labels)
            total_count = len(l_labels)
            uniques, counts = np.unique(l_labels, return_counts=True)
            label_probs = dict(zip(uniques, counts / total_count))

            dominant_prob = float(np.max(counts) / total_count)
            total_purity_sum += dominant_prob

            leaf_purity_info[l_id] = {
                'count': total_count,
                'label_probs': label_probs,
                'purity': dominant_prob,
                'num_unique_labels': len(uniques)
            }

        avg_purity = total_purity_sum / n_leaves if n_leaves > 0 else 0.0
        print(f"Average Leaf Purity: {avg_purity:.4f} across {n_leaves} non-empty leaves.")
        return avg_purity, leaf_purity_info


    def get_depth(self):
        def recurse(node,depth):
            if "value" in node:
                return depth
            else:
                left_depth = recurse(node["left"], depth+1)
                right_depth = recurse(node["right"], depth+1)
                return max(left_depth, right_depth)
        return recurse(self.tree, 0)

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

    def remap_tree(self, feature_history):
        def recurse(node):
            if "value" in node:
                return
            else:
                feat_id, split_val = feature_history.get_feat_split_result(node['feat'])
                node['feat'] = feat_id
                node['split_val'] = split_val
                recurse(node["left"])
                recurse(node["right"])
        recurse(self.tree)

    def extend_tree(self, subtree, leaf_id):
        def merge(node):
            if "value" in node:
                l_id = node["value"]["leaf_id"]
                if leaf_id == l_id:
                    print('MERGED!')
                    return subtree
                return node

            if "left" in node: node["left"] = merge(node["left"])
            if "right" in node: node["right"] = merge(node["right"])
            return node

        self.tree = merge(self.tree)



    def get_intervals_each_path(self, feat_history):
        paths = self.get_all_paths()
        interval_path_dic = {}
        for i, path in enumerate(paths):
            interval_path_dic[i] = calc_intervals_of_path(path['path'], feat_history)
        return interval_path_dic

    def get_all_paths(self): #can be used after remapping tree
        all_paths = []

        def path_finder(node, current_path):
            #leaf
            if 'value' in node:
                all_paths.append(
                    {'path': current_path,
                    'error': node.get('error', 0),
                    'sample_ids': node['value']['sample_ids'],
                    'rel_prob': node['value']['rel_prob']})
                return

            feat_id = int(node['feat'])
            split_val = node['split_val']


            if 'left' in node:
                left_path = {k: (v[0].copy(), v[1].copy()) for k, v in current_path.items()}
                if feat_id not in left_path:
                    left_path[feat_id] = ([], [])
                left_path[feat_id][0].append(split_val)
                left_path[feat_id][1].append('L')
                path_finder(node['left'], left_path)

            if 'right' in node:
                right_path = {k: (v[0].copy(), v[1].copy()) for k, v in current_path.items()}
                if feat_id not in right_path:
                    right_path[feat_id] = ([], [])
                right_path[feat_id][0].append(split_val)
                right_path[feat_id][1].append('R')
                path_finder(node['right'], right_path)

        path_finder(self.tree, {})
        if len(all_paths) != len(self.get_leafs()):
            print('LEN PATHS AND LEAVES SHOULD BE THE SAME')
        return all_paths


def calc_intervals_of_path(path, feat_history):
    feat_interval_dic = {}
    for feature_id in range(len(feat_history.feature_info_list)):
        intervals = Intervals(feat_id=feature_id, chunkinfo=feat_history.chunkInfo)
        feat_interval_dic[feature_id] = add_intervals_for_feat(path.get(feature_id), intervals)
    return feat_interval_dic

def add_intervals_for_feat(path, start_interval):
    if path is None:
        return start_interval
    intervals = start_interval
    for i in range(len(path[0])):
        if path[1][i] == 'R':
            add_right_interval(path[0][i], intervals)
        else:
            add_left_interval(path[0][i], intervals)
    return intervals

def parse_values(value_str: str) -> List[float]:
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", value_str)
    return [float(x) for x in matches]

def add_left_interval(value_str: str, intervals: Intervals):
    #is the true branch
    vals = parse_values(value_str)
    max_val = intervals.max_val
    min_val = intervals.min_val
    if not vals:
        return
    if 'bigger_eq' in value_str:
        intervals.add_interval(vals[0], max_val, 'closed')
    elif 'smaller_eq' in value_str:
        intervals.add_interval(min_val, vals[0], 'closed')
    elif 'even' in value_str:
        intervals.add_interval(vals[0], vals[0], 'closed')
    elif 'interval' in value_str and len(vals) >= 2:
        intervals.add_interval(vals[0], vals[1], 'closed')
    else:
        print('Invalid key or missing values')

def add_right_interval(value_str: str, intervals: Intervals):
    #is the false branch
    vals = parse_values(value_str)
    max_val = intervals.max_val
    min_val = intervals.min_val
    if not vals:
        return
    if 'bigger_eq' in value_str:
        intervals.add_interval(min_val, vals[0], 'half-closed')
    elif 'smaller_eq' in value_str:
        intervals.add_interval(vals[0], max_val, 'half-open')
    elif 'even' in value_str:
        intervals.add_interval(vals[0], vals[0], 'open')
    elif 'interval' in value_str and len(vals) >= 2:
        intervals.add_interval(min_val, vals[0], 'half-closed')
        intervals.add_interval(vals[1], max_val, 'half-open')
    else:
        print('Invalid key or missing values')

def remap_tree(tree,feature_history):
    def recurse(node):
        if "value" in node:
            return
        else:
            feat_id, split_val = feature_history.get_feat_split_result(node['feat'])
            node['feat'] = feat_id
            node['split_val'] = split_val
            recurse(node["left"])
            recurse(node["right"])

    recurse(tree)
    return tree

def extend_tree(tree, hist_tree, leaf_id):
    def merge(node):
        if "value" in node:
            l_id = node["value"]["leaf_id"]
            if leaf_id == l_id:
                print('MERGED!')
                return hist_tree
            return node

        if "left" in node: node["left"] = merge(node["left"])
        if "right" in node: node["right"] = merge(node["right"])
        return node

    return merge(tree)