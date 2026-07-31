from concurrent.futures import ProcessPoolExecutor, as_completed, FIRST_COMPLETED, wait
from src.data.data_obj.feature_history import extend_history
import numpy as np
from src.usePydl.predictors.predictor import Predictor

MIN_NUM_SAMPLES = 4
DEPTH_ITERATIONS= 20
MAX_TIME = 20
EPSILON = 1e-9


def generate_pred(feature_history) -> LocalGreedyPredictor:
    pred = LocalGreedyPredictor(feature_history)
    return pred


class LocalGreedyPredictor(Predictor):
    def __init__(self,feature_history):
        print(f"Starting Greedy Predictor, samples:{feature_history.samples.shape}")
        super().__init__(feature_history=feature_history, max_depth=2, min_sup=1, time=MAX_TIME)
        print(f"Finished Predictor, samples:{feature_history.samples.shape}")


def build_tree_iteratively(feature_history):
    root_predictor = LocalGreedyPredictor(feature_history)

    def get_map(_ids):
        return {j: jd for j, jd in enumerate(_ids)}

    def should_expand(leaf_error, predictor_error):
        return (leaf_error - predictor_error) > EPSILON

    future_to_leaf = {}
    executor = ProcessPoolExecutor(max_workers=6)
    root_history = feature_history
    root_history.tree = root_predictor.tree
    #first root pass ===================================
    leafs = root_predictor.tree.get_leafs()

    for leaf in leafs:
        sample_ids = leaf["value"].get("sample_ids", [])
        leaf_error = leaf.get("error", 0)
        if len(sample_ids) >= MIN_NUM_SAMPLES and len(sample_ids) != feature_history.samples.shape[0]:
            if should_expand(leaf_error, root_predictor.error):
                sub_samples = feature_history.samples[sample_ids]
                new_history = extend_history(sub_samples, root_history, leaf["value"]["leaf_id"])
                new_history.creat_splits()
                future = executor.submit(generate_pred, new_history)
                future_to_leaf[future] = new_history

    while future_to_leaf:
        done, not_done = wait(future_to_leaf.keys(), return_when=FIRST_COMPLETED)

        for future in done:
            current_history = future_to_leaf.pop(future)
            try:
                new_predictor = future.result()
            except Exception as exc:
                print(f"EnsemblePredictor child generated an exception: {exc}")
                continue
            new_tree = new_predictor.tree
            current_history.tree = new_tree
            if current_history.depth < DEPTH_ITERATIONS:
                new_leafs = new_tree.get_leafs()
                for leaf in new_leafs:
                    child_sample_ids = leaf["value"].get("sample_ids", [])
                    child_leaf_error = leaf.get("error", 0)
                    if len(child_sample_ids) >= MIN_NUM_SAMPLES:
                        if should_expand(child_leaf_error, new_predictor.error):
                            child_sub_samples = current_history.samples[child_sample_ids]
                            new_history = extend_history(child_sub_samples, current_history, leaf["value"]["leaf_id"])
                            new_history.creat_splits()
                            future = executor.submit(generate_pred, new_history)
                            future_to_leaf[future] = new_history
    executor.shutdown()
    return root_predictor