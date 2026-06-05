import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import numpy as np

from src.usePydl.leaf import get_leafs
from src.usePydl.predictors.predictor import Predictor

MIN_NUM_SAMPLES = 4
MAX_DEPTH = 20
MAX_TIME = 20
EPSILON = 1e-9


def generate_pred_with_new_process(sub_splits, sub_samples, depth, n_samples, global_feat_map):
    pred = EnsemblePredictor(
        splits=sub_splits,
        samples=sub_samples,
        depth=depth,
        n_samples=n_samples,
        global_feat_map=global_feat_map,
    )
    leafs = get_leafs(pred.get_dl_tree())
    return pred, leafs, sub_samples.shape[0]


class EnsemblePredictor(Predictor):
    def __init__(self, splits, samples, depth=0, n_samples=None, global_feat_map=None):
        self.depth = depth

        if n_samples is not None:
            self.n_samples = n_samples
        else:
            self.n_samples = samples.shape[0]

        if global_feat_map is None:
            self.global_feat_map = np.arange(splits.shape[1])
        else:
            self.global_feat_map = np.asarray(global_feat_map)

        self.child_predictors_dic = {}

        print(f"Starting Ensemble Predictor depth:{self.depth}, samples:{samples.shape}")

        super().__init__(
            splits=splits,
            samples=samples,
            max_depth=2,
            min_sup=1,
            time=MAX_TIME,
            n_samples=self.n_samples,
        )

        print(f"Finished Predictor depth:{self.depth}, samples:{samples.shape}")

    def gen_new_data(self, splits=None, feature_info=None, samples=None, n_new_samples: int = 100,
                     conf_tresh: float = 0.8) -> np.ndarray:
        tree = self.get_complete_tree()
        return self.gen_new_data_based_on_tree(
            tree=tree,
            splits=splits,
            feature_info=feature_info,
            n_new_samples=n_new_samples,
            conf_tresh=conf_tresh,
            old_samples=samples
        )

    def get_complete_tree(self):
        local_tree = copy.deepcopy(self.get_dl_tree())

        def remap_features(node):
            if not isinstance(node, dict):
                return
            if "feat" in node:
                node["feat"] = int(self.global_feat_map[node["feat"]])
            if "left" in node:
                remap_features(node["left"])
            if "right" in node:
                remap_features(node["right"])

        remap_features(local_tree)

        if not self.child_predictors_dic:
            return local_tree

        for leaf_signature, pred in self.child_predictors_dic.items():
            pred_tree = pred.get_complete_tree()
            local_tree = merge_subtrees_into_parent(
                local_tree,
                copy.deepcopy(pred_tree),
                leaf_signature,
            )

        return local_tree


def leaf_signature_from_leaf(leaf):
    sample_ids = tuple(sorted(leaf.get("value", {}).get("sample_ids", [])))
    return sample_ids


def merge_subtrees_into_parent(tree, subtree, target_signature):
    replaced = False

    def merge(node):
        nonlocal replaced
        if replaced:
            return node

        if "value" in node:
            sig = tuple(sorted(node["value"].get("sample_ids", [])))
            if sig == target_signature:
                replaced = True
                return subtree
            return node

        if "left" in node:
            node["left"] = merge(node["left"])
        if "right" in node:
            node["right"] = merge(node["right"])

        return node

    return merge(tree)


def should_expand(leaf_error, predictor_error):
    return (leaf_error - predictor_error) > EPSILON


def build_ensemble_tree_iteratively(splits, samples):
    root_predictor = EnsemblePredictor(
        splits=splits,
        samples=samples,
        n_samples=samples.shape[0],
        depth=0,
    )
    current_level_tasks = []
    initial_leafs = get_leafs(root_predictor.get_dl_tree())

    for leaf in initial_leafs:
        sample_ids = leaf["value"].get("sample_ids", [])
        leaf_error = leaf.get("error", 0)

        if (len(sample_ids) >= MIN_NUM_SAMPLES and len(sample_ids) != samples.shape[0]):
            if should_expand(leaf_error, root_predictor.error):
                current_level_tasks.append((root_predictor, leaf_signature_from_leaf(leaf), sample_ids))

    with ProcessPoolExecutor(max_workers=6) as executor:
        while current_level_tasks:
            future_to_meta = {}

            for parent, leaf_signature, s_ids in current_level_tasks:
                if parent.depth >= MAX_DEPTH:
                    continue

                sub_samples = samples[s_ids, :]
                sub_splits = splits[s_ids, :]

                # Homogeneous split pruning
                col_sums = sub_splits.sum(axis=0)
                mask = (col_sums > 0) & (col_sums < len(s_ids))

                if not np.any(mask):
                    continue

                filtered_sub_splits = sub_splits[:, mask]

                # FIX: Since `splits` here is the absolute GLOBAL splits array, 
                # evaluating the mask directly yields the absolute global indices!
                global_indices = np.where(mask)[0]

                future = executor.submit(
                    generate_pred_with_new_process,
                    filtered_sub_splits,
                    sub_samples,
                    parent.depth + 1,
                    len(s_ids),
                    global_indices,
                )

                future_to_meta[future] = (parent, leaf_signature, s_ids)

            next_level_tasks = []

            for future in as_completed(future_to_meta):
                parent, leaf_signature, s_ids = future_to_meta[future]
                try:
                    child_predictor, child_leafs, parent_subset_size = future.result()
                    parent.child_predictors_dic[leaf_signature] = child_predictor

                    for child_leaf in child_leafs:
                        local_ids = child_leaf["value"].get("sample_ids", [])
                        child_error = child_leaf.get("error", 0)

                        global_sample_ids = [s_ids[i] for i in local_ids]

                        if len(global_sample_ids) < MIN_NUM_SAMPLES:
                            continue
                        if len(global_sample_ids) == parent_subset_size:
                            continue
                        if not should_expand(child_error, child_predictor.error):
                            continue

                        next_level_tasks.append(
                            (
                                child_predictor,
                                tuple(sorted(global_sample_ids)),
                                global_sample_ids,
                            )
                        )

                except Exception as e:
                    print(f"Error expanding leaf: {e}")

            current_level_tasks = next_level_tasks

    return root_predictor