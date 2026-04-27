import numpy as np
from typing import Dict, Any, Optional, List


def leaf_val(predictor, samples):
    def value(tids):
        features = np.array(samples[list(tids)]).T
        print(features.shape)
        distributions =predictor.get_distributions(features)
        return {"count":len(features),"distr":distributions}
    return value

def empty_leave_val():
    def value(tids):
        return 1
    return value


def get_leaf_vals(tree):
    leaves = []

    def recurse(node):
        if "value" in node:
            leaves.append(node)
        else:
            recurse(node["left"])
            recurse(node["right"])

    recurse(tree)
    return leaves


class VizTree:
    def __init__(self, tree_dict: Optional[Dict[str, Any]] = None):
        self.tree_dict = tree_dict
        self.print_tree()

    def print_tree(self, max_depth: Optional[int] = None, show_feature_names: Optional[List[str]] = None):
        if self.tree_dict is None:
            print("No tree to display.")
            return

        lines = self._build_tree_lines(self.tree_dict, max_depth=max_depth,feature_names=show_feature_names)
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



