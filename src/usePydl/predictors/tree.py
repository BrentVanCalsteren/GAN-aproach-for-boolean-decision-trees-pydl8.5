import re
from typing import Dict, List
from src.usePydl.predictors.interval import Intervals


class Tree:

    def __init__(self, tree):
        self.tree : Dict = tree

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


    def get_intervals_each_path(self, feat_len: int):
        paths = self.get_all_paths()
        interval_path_dic = {}
        for i, path in enumerate(paths):
            interval_path_dic[i] = calc_intervals_of_path(path['path'], feat_len)
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


def calc_intervals_of_path(path, num_feat_len):
    feat_interval_dic = {}
    for feature_id in range(num_feat_len):
        intervals = Intervals()
        feat_interval_dic[feature_id] = add_intervals(path.get(feature_id), intervals)
    return feat_interval_dic

def add_intervals(path, start_interval):
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
    if not vals:
        return
    if 'bigger_eq' in value_str:
        intervals.add_interval(vals[0], 1, 'closed')
    elif 'smaller_eq' in value_str:
        intervals.add_interval(0, vals[0], 'closed')
    elif 'even' in value_str:
        intervals.add_interval(vals[0], vals[0], 'closed')
    elif 'interval' in value_str and len(vals) >= 2:
        intervals.add_interval(vals[0], vals[1], 'closed')
    else:
        print('Invalid key or missing values')

def add_right_interval(value_str: str, intervals: Intervals):
    #is the false branch
    vals = parse_values(value_str)
    if not vals:
        return
    if 'bigger_eq' in value_str:
        intervals.add_interval(0, vals[0], 'half-closed')
    elif 'smaller_eq' in value_str:
        intervals.add_interval(vals[0], 1, 'half-open')
    elif 'even' in value_str:
        intervals.add_interval(vals[0], vals[0], 'open')
    elif 'interval' in value_str and len(vals) >= 2:
        intervals.add_interval(0, vals[0], 'half-closed')
        intervals.add_interval(vals[1], 1, 'half-open')
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