import numpy as np
from src.samplers.load_samplers import create_sampler
#default leaf value -> return the samplers
def default_leaf_val(samples, sampler_types):
    def value(tids):
        features = np.array(samples[list(tids)]).T
        samplers = []
        for i, feat in enumerate(features):
            sampler = create_sampler(sampler_types[i])
            sampler.fit(feat)
            samplers.append(sampler)
        return {"count":len(list(tids)),
                "samplers":samplers,
                "sample_ids": list(tids)}
    return value


#other leaf val functions: NOT USED RIGHT NOW
def empty_val():
    def value(_):
        return 1
    return value
#-------------------------------------------------

#helpers for looking at leaf data
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

#viz tree in console
class VizTree:
    def __init__(self, tree_dict=None):
        self.tree_dict = tree_dict
        self.print_tree()

    def print_tree(self, max_depth=None, feature_names=None):
        print(self.get_tree_string(max_depth, feature_names))

    def get_tree_string(self, max_depth=None, feature_names=None):
        if self.tree_dict is None:
            return "No tree to display."
        lines = []
        self._build(self.tree_dict, lines, max_depth, feature_names)
        return "\n".join(lines)

    def _build(self, node, lines, max_depth=None, feature_names=None, depth=0, prefix=""):
        if max_depth is not None and depth > max_depth:
            lines.append(prefix + "...")
            return

        if "value" in node:
            label = f"Class: {node['value']}"
        elif "feat" in node:
            feat = node["feat"]
            label = feature_names[feat] if feature_names else f"feat {feat}"
        else:
            label = "?"

        lines.append(prefix + label)

        for key, symbol in [("left", "L"), ("right", "R")]:
            if node.get(key):
                self._build(node[key], lines, max_depth, feature_names, depth + 1, prefix + f"  {symbol} ")

    def __str__(self):
        return self.get_tree_string()



