from src.usePydl.predictors.predictor import Predictor
from src.usePydl.leaf import get_leafs
import numpy as np
import concurrent.futures
import copy

MIN_NUM_SAMPLES = 10


class EnsemblePredictor(Predictor):
    def __init__(self,splits, samples,depth=0, n_samples=None):
        self.depth = depth
        if n_samples:
            self.n_samples = n_samples
        else:
            self.n_samples = samples.shape[0]
        print(f"Starting Ensemble Predictor, depth:{self.depth}, samples:{samples.shape} ")
        self.low_error = 0.01*samples.shape[1]
        self.child_predictors_dic = {}
        super().__init__(
            splits=splits,
            samples=samples,
            max_depth=2,
            min_sup=1,
            time=10,
            n_samples=n_samples
        )


    def gen_new_data(self, splits=None, feature_info=None,samples=None,n_new_samples: int = 100, conf_tresh: float = 0.8) -> np.ndarray:
        tree = self.get_complete_tree()
        return self.gen_new_data_based_on_tree(tree=tree,
                                        splits=splits,
                                        feature_info=feature_info,
                                        n_new_samples=n_new_samples,
                                        conf_tresh=conf_tresh,
                                        old_samples=samples)


    def get_complete_tree(self):
        local_tree = self.get_dl_tree()
        if self.child_predictors_dic:
            combo_tree = copy.deepcopy(local_tree)
            orig_leafs = get_leafs(local_tree)
            for i,pred in self.child_predictors_dic.items():
                pred_tree = pred.get_complete_tree()
                target_ids = set(orig_leafs[i]["value"].get('sample_ids', []))
                combo_tree = merge_subtrees_into_parent(combo_tree, copy.deepcopy(pred_tree), target_ids)
            return combo_tree
        else: return copy.deepcopy(local_tree)


def merge_subtrees_into_parent(tree, subtree, target_sample_ids):
    def merge(node):
        if 'value' in node:
            leaf_ids = set(node['value'].get("sample_ids", []))
            if leaf_ids == target_sample_ids:
                return subtree
            return node
        if 'left' in node:
            node['left'] = merge(node['left'])

        if 'right' in node:
            node['right'] = merge(node['right'])
        return node
    return merge(tree)

def build_ensemble_tree_iteratively(splits, samples):
    root_predictor = EnsemblePredictor(splits=splits, samples=samples, n_samples=samples.shape[0],depth=0)
    queue = []

    initial_leafs = get_leafs(root_predictor.get_dl_tree())
    for i, leaf in enumerate(initial_leafs):
        sample_ids = leaf["value"].get("sample_ids", [])
        leaf_error = leaf.get("error", 0)
        if (len(sample_ids) >= MIN_NUM_SAMPLES and len(sample_ids) != samples.shape[0] and leaf_error >= root_predictor.low_error):
            queue.append((root_predictor, i, sample_ids, 0))

    while len(queue) > 0:
        parent_predictor, leaf_idx, sample_ids, current_depth = queue.pop(0)
        sub_samples = samples[sample_ids, :]
        sub_splits = splits[sample_ids, :]

        print(f"Expanding Leaf {leaf_idx} at Depth {current_depth} with {len(sample_ids)} samples")

        child_predictor = EnsemblePredictor(splits=sub_splits,samples=sub_samples,depth=current_depth + 1,n_samples=root_predictor.n_samples)
        parent_predictor.child_predictors_dic[leaf_idx] = child_predictor
        child_leafs = get_leafs(child_predictor.get_dl_tree())
        for next_i, child_leaf in enumerate(child_leafs):
            child_sample_ids = child_leaf["value"].get("sample_ids", [])
            child_leaf_error = child_leaf.get("error", 0)
            if (len(child_sample_ids) >= MIN_NUM_SAMPLES and len(child_sample_ids) != sub_samples.shape[0] and child_leaf_error >= child_predictor.low_error):
                queue.append((child_predictor, next_i, child_sample_ids, current_depth + 1))

    return root_predictor