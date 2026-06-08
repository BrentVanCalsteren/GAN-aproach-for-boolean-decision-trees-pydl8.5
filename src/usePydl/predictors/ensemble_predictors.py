from concurrent.futures import ProcessPoolExecutor, as_completed, FIRST_COMPLETED, wait
import numpy as np
from src.usePydl.predictors.predictor import Predictor

MIN_NUM_SAMPLES = 4
MAX_DEPTH = 20
MAX_TIME = 20
EPSILON = 1e-9


def generate_pred(sub_splits, sub_samples, n_samples) -> EnsemblePredictor:
    pred = EnsemblePredictor(
        splits=sub_splits,
        samples=sub_samples,
        n_samples=n_samples,
    )
    return pred


class EnsemblePredictor(Predictor):
    def __init__(self,samples, splits=None, n_samples=None):
        print(f"Starting Ensemble Predictor, samples:{samples.shape}")

        super().__init__(splits=splits, samples=samples, max_depth=2, min_sup=1, time=MAX_TIME, n_samples=n_samples)

        print(f"Finished Predictor, samples:{samples.shape}")


def build_ensembles_iteratively(splits, samples):
    root_predictor = EnsemblePredictor(samples=samples, splits=splits, n_samples=samples.shape[0])

    def get_map(_ids):
        return {j: jd for j, jd in enumerate(_ids)}

    def should_expand(leaf_error, predictor_error):
        return (leaf_error - predictor_error) > EPSILON

    future_to_leaf = {}
    executor = ProcessPoolExecutor(max_workers=6)

    leafs = root_predictor.tree.get_leafs()
    for leaf in leafs:
        sample_ids = leaf["value"].get("sample_ids", [])
        leaf_error = leaf.get("error", 0)
        leaf_signature = tuple(sorted(sample_ids))

        if len(sample_ids) >= MIN_NUM_SAMPLES and len(sample_ids) != samples.shape[0]:
            if should_expand(leaf_error, root_predictor.error):
                sample_map = get_map(sample_ids)
                sub_samples = samples[sample_ids]

                sub_splits = splits[sample_ids]
                col_sums = sub_splits.sum(axis=0)
                mask = (col_sums > 0) & (col_sums < len(sample_ids))
                if not np.any(mask):
                    continue
                sub_splits = sub_splits[:, mask]
                global_splits = np.where(mask)[0]
                split_map = get_map(global_splits)

                future = executor.submit(generate_pred, sub_splits, sub_samples, samples.shape[0])
                future_to_leaf[future] = (leaf_signature, split_map, sample_map, 1)

    while future_to_leaf:
        done, not_done = wait(future_to_leaf.keys(), return_when=FIRST_COMPLETED)

        for future in done:
            leaf_signature, split_map, sample_map, depth = future_to_leaf.pop(future)
            try:
                new_predictor = future.result()
            except Exception as exc:
                print(f"EnsemblePredictor child generated an exception: {exc}")
                continue

            new_tree = new_predictor.tree
            new_tree.remap_tree(split_map, sample_map)
            root_predictor.tree.extend_tree(new_tree.tree, leaf_signature)

            if depth < MAX_DEPTH:
                new_leafs = new_tree.get_leafs()
                for leaf in new_leafs:
                    child_sample_ids = leaf["value"].get("sample_ids", [])
                    child_leaf_error = leaf.get("error", 0)
                    child_signature = tuple(sorted(child_sample_ids))

                    if len(child_sample_ids) >= MIN_NUM_SAMPLES:
                        if should_expand(child_leaf_error, root_predictor.error):
                            child_sample_map = get_map(child_sample_ids)
                            child_sub_samples = samples[child_sample_ids]

                            child_sub_splits = splits[child_sample_ids]
                            col_sums = child_sub_splits.sum(axis=0)
                            mask = (col_sums > 0) & (col_sums < len(child_sample_ids))
                            if not np.any(mask):
                                continue
                            child_sub_splits = child_sub_splits[:, mask]
                            child_global_splits = np.where(mask)[0]
                            child_split_map = get_map(child_global_splits)

                            new_future = executor.submit(generate_pred, child_sub_splits, child_sub_samples,
                                                         samples.shape[0])
                            future_to_leaf[new_future] = (child_signature, child_split_map, child_sample_map, depth + 1)

    executor.shutdown()
    return root_predictor