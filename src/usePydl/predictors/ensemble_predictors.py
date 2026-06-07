from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
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
    leafs = pred.tree.get_leafs()
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
            splits_obj=splits,
            samples=samples,
            max_depth=2,
            min_sup=1,
            time=MAX_TIME,
            n_samples=self.n_samples,
        )

        print(f"Finished Predictor depth:{self.depth}, samples:{samples.shape}")


def leaf_signature_from_leaf(leaf):
    sample_ids = tuple(sorted(leaf.get("value", {}).get("sample_ids", [])))
    return sample_ids


def should_expand(leaf_error, predictor_error):
    return (leaf_error - predictor_error) > EPSILON


def build_ensembles_iteratively(splits, samples):
    root_predictor = EnsemblePredictor(
        splits=splits,
        samples=samples,
        n_samples=samples.shape[0],
        depth=0,
    )
    current_level_tasks = []
    initial_leafs = root_predictor.tree.get_leafs()

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