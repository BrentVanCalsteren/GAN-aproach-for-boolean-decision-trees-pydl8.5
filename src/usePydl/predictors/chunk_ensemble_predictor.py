from typing import List
import numpy as np
from src.data.data_obj.feature_history import FeatureHistory
from src.usePydl.predictors.local_greedy_predictors import build_tree_iteratively, LocalGreedyPredictor
from src.usePydl.predictors.predictor import Predictor


class ChunkEnsemblePredictor:
    def __init__(self):
        self.predictors: List[Predictor] = []
        self.chunk_sizes: List[int] = []

    def add_chunk(self, feature_history: FeatureHistory):
        print(f"--- Training ChunkEnsemblePredictor on chunk with {feature_history.samples.shape[0]} samples ---")
        feature_history.creat_splits()
        predictor = build_tree_iteratively(feature_history)
        self.predictors.append(predictor)
        self.chunk_sizes.append(feature_history.samples.shape[0])

    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if len(self.predictors) == 0:
            raise ValueError("No chunk predictors in ensemble. Call add_chunk() first.")
        total_samples = sum(self.chunk_sizes)
        if total_samples == 0:
            weights = np.ones(len(self.predictors)) / len(self.predictors)
        else:
            weights = np.array(self.chunk_sizes, dtype=float) / total_samples
        # Distribute sample count n across predictors via multinomial
        counts = np.random.multinomial(n, weights)
        chunk_samples_list = []
        for i, predictor in enumerate(self.predictors):
            chunk_n = counts[i]
            if chunk_n == 0:
                continue
            samples_k = predictor.gen_new_data_based_tree_structure(n=chunk_n, conf=conf)
            if samples_k.size > 0:
                chunk_samples_list.append(samples_k)
        if not chunk_samples_list:
            return np.array([])
        combined_samples = np.vstack(chunk_samples_list)
        np.random.shuffle(combined_samples)
        return combined_samples[:n]