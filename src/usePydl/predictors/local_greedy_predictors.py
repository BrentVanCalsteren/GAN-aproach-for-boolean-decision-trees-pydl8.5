from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from src.data.data_obj.feature_history import extend_history
import numpy as np
from src.usePydl.predictors.predictor import Predictor
import CONFIG

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
    feature_history.tree = root_predictor.tree
    future_to_leaf = {}

    def submit_leaves(history, predictor_error, total_samples=None):
        for leaf in history.tree.get_leafs():
            sample_ids = leaf["value"].get("sample_ids", [])
            leaf_error = leaf.get("error", 0)

            if total_samples and len(sample_ids) == total_samples:
                continue

            if len(sample_ids) >= CONFIG.MIN_SAMPLES_IN_LEAF and (leaf_error - predictor_error) > EPSILON:
                sub_history = extend_history(history.samples[sample_ids], history, leaf["value"]["leaf_id"])
                sub_history.creat_splits()
                future = executor.submit(generate_pred, sub_history)
                future_to_leaf[future] = sub_history

    with ProcessPoolExecutor(max_workers=6) as executor:
        submit_leaves(feature_history, root_predictor.error, total_samples=feature_history.samples.shape[0])
        while future_to_leaf:
            done, _ = wait(future_to_leaf.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                current_history = future_to_leaf.pop(future)
                try:
                    new_pred = future.result()
                except Exception as exc:
                    print(f"Child predictor generated an exception: {exc}")
                    continue

                current_history.tree = new_pred.tree

                if current_history.depth < CONFIG.MAX_GREEDY_DEPTH:
                    submit_leaves(current_history, new_pred.error)

    return root_predictor