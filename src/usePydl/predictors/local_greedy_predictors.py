from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from src.data.data_obj.feature_history import extend_history
import numpy as np
from src.usePydl.predictors.predictor import Predictor
import CONFIG

MAX_TIME = 20
EPSILON = 1e-9


def generate_pred(feature_history, weights=None) -> LocalGreedyPredictor:
    pred = LocalGreedyPredictor(feature_history, weights)
    return pred


class LocalGreedyPredictor(Predictor):
    def __init__(self,feature_history,weights=None):
        print(f"Starting Greedy Predictor, samples:{feature_history.samples.shape}")
        super().__init__(feat_hist=feature_history,weights=weights, max_depth=2, min_sup=1, time=CONFIG.MAX_TIME_PREDICTOR)
        print(f"Finished Predictor, samples:{feature_history.samples.shape}")


def build_tree_iteratively(feature_history, weights=None, n_worksers=10):
    root_predictor = LocalGreedyPredictor(feature_history)
    feature_history.tree = root_predictor.tree
    future_to_leaf = {}

    def submit_leaves(history):
        print(f"Submitting leaves, depth:{history.depth}")
        for leaf in history.tree.get_leafs():
            sample_ids = leaf["value"].get("sample_ids", [])

            if len(sample_ids) >= CONFIG.MIN_SAMPLES_IN_LEAF:
                sub_history = extend_history(history.samples[sample_ids], history, leaf["value"]["leaf_id"])
                if weights is None:
                    if sub_history.depth < 2: new_weights = sub_history.get_feature_weights(mode='uniform',focus_on=1.0)
                    else: new_weights = sub_history.get_feature_weights(mode='random',focus_on=1.0)
                else: new_weights = weights
                sub_history.creat_splits(weight_of_each_feature=new_weights)
                future = executor.submit(generate_pred, sub_history, new_weights)
                future_to_leaf[future] = sub_history

    with ProcessPoolExecutor(max_workers=n_worksers) as executor:
        submit_leaves(feature_history)
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
                current_history.pred_error = new_pred.error
                if current_history.past is not None:
                    past_error = current_history.past.pred_error
                else: past_error = np.inf

                if current_history.depth < CONFIG.MAX_GREEDY_DEPTH and (
                        past_error == np.inf or
                        (past_error - new_pred.error) > past_error/100):
                    print(f'error reduction! : {past_error - new_pred.error}')
                    submit_leaves(current_history)

    return root_predictor
