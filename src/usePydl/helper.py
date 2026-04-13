from typing import Dict, Any, Optional, List
import random

import numpy as np

import src.dataLoader.dataset_loader as loader
import src.binaryConvertion.binner as binner
from sklearn.model_selection import train_test_split

class VizTree:
    """
    Console-based visualizer based on decision trees representation in pydl8.5 .tree_ obj.
    """

    def __init__(self, tree_dict: Optional[Dict[str, Any]] = None):
        self.tree_dict = tree_dict
        self.print_tree()

    def print_tree(self, max_depth: Optional[int] = None, show_feature_names: Optional[List[str]] = None):
        if self.tree_dict is None:
            print("No tree to display.")
            return

        lines = self._build_tree_lines(self.tree_dict, max_depth=max_depth,
                                       feature_names=show_feature_names)
        for line in lines:
            print(line)

    def get_tree_string(self, max_depth: Optional[int] = None,
                        feature_names: Optional[List[str]] = None) -> str:
        if self.tree_dict is None:
            return "No tree to display."
        lines = self._build_tree_lines(self.tree_dict, max_depth, feature_names)
        return "\n".join(lines)

    def _build_tree_lines(self, node: Dict[str, Any], depth: int = 0,
                          prefix: str = "", is_left: bool = None,
                          max_depth: Optional[int] = None,
                          feature_names: Optional[List[str]] = None) -> List[str]:
        if max_depth is not None and depth > max_depth:
            return [prefix + "└── ... (max depth reached)"]

        lines = []
        if 'value' in node:
            label = f"⚫ Class: {node['value']}"
        elif 'feat' in node:
            feat = node['feat']
            if feature_names and 0 <= feat < len(feature_names):
                feat_str = f"{feature_names[feat]} (feat {feat})"
            else:
                feat_str = f"feature {feat}"
            label = f" {feat_str}"
        else:
            label = "Unknown node"

        # Add current node
        if depth == 0:
            lines.append(label)
        else:
            branch = "└── " if is_left is None else ("├── " if not is_left else "└── ")
            lines.append(prefix + branch + label)

        # Prepare new prefix for children
        if depth == 0:
            new_prefix = ""
        else:
            new_prefix = prefix + ("    " if is_left else "│   ")

        # Process children
        if 'left' in node and node['left']:
            left_lines = self._build_tree_lines(node['left'], depth + 1,
                                                new_prefix, is_left=False,
                                                max_depth=max_depth,
                                                feature_names=feature_names)
            lines.extend(left_lines)
        if 'right' in node and node['right']:
            right_lines = self._build_tree_lines(node['right'], depth + 1,
                                                 new_prefix, is_left=True,
                                                 max_depth=max_depth,
                                                 feature_names=feature_names)
            lines.extend(right_lines)

        return lines

    def depth_first_nodes(self) -> Dict[int, List[str]]:
        if self.tree_dict is None:
            return {}
        depth_dict = {}
        self._collect_nodes_by_depth(self.tree_dict, 0, depth_dict)
        return depth_dict

    def _collect_nodes_by_depth(self, node: Dict[str, Any], depth: int, depth_dict: Dict[int, List[str]]):
        if depth not in depth_dict:
            depth_dict[depth] = []

        if 'value' in node:
            depth_dict[depth].append(f"Class: {node['value']}")
        if 'feat' in node:
            depth_dict[depth].append(f"Feature {node['feat']}")

        if 'left' in node:
            self._collect_nodes_by_depth(node['left'], depth + 1, depth_dict)
        if 'right' in node:
            self._collect_nodes_by_depth(node['right'], depth + 1, depth_dict)

    def print_by_depth(self):
        """Print nodes grouped by depth (legacy style)."""
        depth_dict = self.depth_first_nodes()
        for depth in sorted(depth_dict.keys()):
            print(f"Depth {depth}: {', '.join(depth_dict[depth])}")

    def __str__(self) -> str:
        return self.get_tree_string()


def prep_data_for_pydl_no_sep(dataset_name:str='bank',max_bool_lenght=12):
    dataset = loader.load_dataloader_by_name(dataset_name,y_seperated=False)
    complete_x = dataset.get_x_complete()
    missing_x = dataset.get_x_missing()
    scaled_x_T = loader.standardize_2d_array(complete_x.T)
    bin_string_x, bin_length_x,clusters = binner.bin_convertion_2d(scaled_x_T,max_bins=max_bool_lenght)
    dl_x = np.array([binner.flatten_binary_strings(row) for row in bin_string_x.T])
    scaled_x = scaled_x_T.T
    return complete_x,missing_x,dl_x,scaled_x,bin_length_x,clusters

def randomize_data(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random.randint(1, 100))
    return X_train, X_test, y_train, y_test

def convert_num_specific_bin_length(num_data_T, clusters):
    bin_data = []
    for i in range(len(num_data_T)):
        bin_data.append(binner.gen_one_hot_string(num_data_T[i], clusters[i]))
    print(bin_data)
    return np.array([binner.flatten_binary_strings(row) for row in np.array(bin_data).T])