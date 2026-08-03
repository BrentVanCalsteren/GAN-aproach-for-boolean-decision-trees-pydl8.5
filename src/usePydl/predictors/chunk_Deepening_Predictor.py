from typing import List, Optional
import numpy as np
from src.data.data_obj.feature_history import FeatureHistory, extend_history
from src.usePydl.predictors.local_greedy_predictors import build_tree_iteratively, LocalGreedyPredictor
from src.usePydl.predictors.predictor import Predictor



class IterativeDeepeningPredictor:
    def __init__(self):
        self.master_history: Optional[FeatureHistory] = None
        self.master_predictor: Optional[Predictor] = None
        self.chunk_count: int = 0


    def fit_initial_chunk(self, feature_history: FeatureHistory):
        print(f"new ChunkDeepeningPredictor")
        feature_history.creat_splits()
        self.master_history = feature_history
        self.master_predictor = build_tree_iteratively(feature_history)
        self.chunk_count = 1


    def extend_with_chunk(self, feature_history: FeatureHistory):
        if self.master_history is None or self.master_predictor is None:
            self.fit_initial_chunk(feature_history)
            return
        self.chunk_count += 1
        print(f"extend ChunkDeepeningPredictor with Chunk #{self.chunk_count}")
        feature_history.creat_splits()
        chunk_predictor = build_tree_iteratively(feature_history)
        if chunk_predictor.tree is not None:
            chunk_hist = chunk_predictor.feature_history
            complete_tree = self.master_history.get_complete_tree()
            leafs = complete_tree.get_leafs() if complete_tree else []
            if len(leafs) > 0:
                target_leaf_id = leafs[0]['value'].get('leaf_id')
                chunk_hist.leaf_id = target_leaf_id
                self.master_history.add_future(chunk_hist)


    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if self.master_predictor is None:
            raise ValueError("no predictor found")
        return self.master_predictor.gen_new_data_based_tree_structure(n=n, conf=conf)