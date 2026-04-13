from pydl85 import DL85Predictor
from sklearn.metrics import DistanceMetric
import numpy as np

class pydl_costum:

    features = None
    bin_features = None
    tree_obj = None

    def __init__(self, bin_features, features):
        self.features = features
        self.bin_features = bin_features

    def viz_tree(self):
        print("simple viz -- tree based on depth")
        if self.tree_obj is None:
            print("no tree generated yet")
        depth_dict_tree = self.find_depth_all_nodes(tree_node=self.tree_obj.tree_)
        for i in range(len(depth_dict_tree)):
            print(depth_dict_tree[i])

    def find_depth_all_nodes(self, tree_node, depth=0, depth_dict={}):
        if depth not in depth_dict:
            depth_dict[depth] = []
        if 'value' in tree_node:
            class_val = tree_node['value']
            depth_dict[depth].append("Class: " + str(class_val))
        if 'feat' in tree_node:
            depth_dict[depth].append("feature " + str(tree_node['feat']))
        if 'left' in tree_node:
            self.find_depth_all_nodes(tree_node['left'], depth + 1, depth_dict)
        if 'right' in tree_node:
            self.find_depth_all_nodes(tree_node['right'], depth + 1, depth_dict)
        return depth_dict

    def create_new_tree(self):
        if self.bin_features is None or self.features is None:
            print("there are no features")
            return
        # =================================
        # --------Error_function----------
        # =================================
        def clustering_error_real_vals(tids):
            samples_original = self.features[list(tids)]
            if len(samples_original) == 0:
                return 0.0

            centroid = np.mean(samples_original, axis=0)
            eucl_dist = DistanceMetric.get_metric('euclidean')
            distances = eucl_dist.pairwise(samples_original, [centroid])
            return float(np.sum(distances))

        # =================================
        # --------leaf_functions----------
        # =================================
        def leaf_value(tids):
            samples = self.features[list(tids)]
            if len(samples) == 0:
                return 0
            return np.sum(np.mean(samples, axis=0))


        self.tree_obj = DL85Predictor(max_depth=6, min_sup=1,
            repeat_sort=True,
            error_function=clustering_error_real_vals,
            leaf_value_function=leaf_value,
            time_limit=600
        )