import random

import numpy as np
from typing import List, Dict, Optional

from src.data.data_obj.feature_history import FeatureHistory
from src.samplers.single_gaussian import SingleGaussian1DSampler
from src.usePydl.predictors.local_greedy_predictors import build_tree_iteratively
from src.usePydl.predictors.tree import Tree, calc_intervals_of_path
from src.data.data_obj.sampels import Samples


class RandomForestPredictor:
    def __init__(self, samples_obj: Samples, max_features_per_tree = 5, subsample_ratio: float = 0.5):
        self.ensemble_trees: List[Tree] = []
        self.tree_feat_histories: List[FeatureHistory] = []
        self.samples_obj = samples_obj
        self.chunk_info = samples_obj.chunk_info
        self.n_chunks = samples_obj.loader.n_chunks
        self.n_features = self.samples_obj.samples.shape[1]
        self.features_each_tree = self.n_features
        self.trees_each_chunk = 1
        self.subsample_ratio = subsample_ratio

        feature_factor = int(np.ceil(np.sqrt(self.n_features)))
        n_features_each_tree = int(min(max_features_per_tree, feature_factor  + feature_factor/2))
        if n_features_each_tree > self.n_features:
            self.features_each_tree = int(self.n_features)
            self.trees_each_chunk = 1
        else:
            self.features_each_tree = n_features_each_tree
            self.trees_each_chunk = max(4, int(np.ceil((self.n_features / self.features_each_tree) * 3)))
        self.n_trees = self.n_chunks * self.trees_each_chunk
        print(f"Init Random Forest: {self.n_trees} trees, each {self.features_each_tree} features")
        self.build_ensemble()


    def build_ensemble(self):
        for chunk_id in range(self.n_chunks):
            self.samples_obj.load_chunk(chunk_id)
            chunk_samples = self.samples_obj.samples
            n_total = chunk_samples.shape[0]
            n_sub = int(np.ceil(n_total * self.subsample_ratio))
            sub_indices = np.random.choice(n_total, size=n_sub, replace=True)
            sub_samples = chunk_samples[sub_indices]
            feature_groups = create_complete_random_feat_groups(n_groups=self.trees_each_chunk,
                                                                features_each_group=self.features_each_tree,
                                                                total_features=self.n_features)
            for group in feature_groups:
                print(f"Tree(chunk:{chunk_id}, feat:{group}) TRAINING")
                feat_hist = FeatureHistory(sub_samples, self.samples_obj.chunk_info)
                weights = feat_hist.get_feature_weights(mode='uniform', focus_on=group)
                feat_hist.creat_splits(weight_of_each_feature=weights)
                pred = build_tree_iteratively(feat_hist, weights)
                complete_tree = feat_hist.get_complete_tree()

                if complete_tree is None or complete_tree.tree is None:
                    complete_tree = pred.tree

                complete_tree.remove_sample_ids_from_leafs()
                feat_hist.reduce_memory()
                self.ensemble_trees.append(complete_tree)
                self.tree_feat_histories.append(feat_hist)
                print(f"Tree(chunk:{chunk_id}, feat:{group}) Trained")

    def gen_new_data(self, n: int = 100, reference_samples: Optional[np.ndarray] = None, conf: float = 0.8):
        if reference_samples is None:
            reference_samples = self.samples_obj.samples

        n_ref = reference_samples.shape[0]
        indices_ref = np.random.choice(n_ref, size=n, replace=True)
        gen_samples = np.zeros((n, self.n_features))

        for i, ref_id in enumerate(indices_ref):
            ref_sample = reference_samples[ref_id]
            tree_candidates = []
            tree_scores = []

            for t_idx, tree in enumerate(self.ensemble_trees):
                feat_hist_t = self.tree_feat_histories[t_idx]
                res = tree.get_leaf_id_for_sample(ref_sample)
                leaf_id, path_dict = res[0], res[1]
                path_error = res[2] if len(res) > 2 else 0.0

                interval_dic = calc_intervals_of_path(path_dict, feat_hist_t)
                candidate_vector = np.zeros(self.n_features)
                scores_vector = np.zeros(self.n_features)

                for f_id in range(self.n_features):
                    n_splits = len(path_dict[f_id][0]) if f_id in path_dict else 0
                    interval_obj = interval_dic[f_id]
                    domain = interval_obj.get_complete_domain()
                    min_v = domain[0]
                    max_v = domain[1]
                    span = max_v - min_v

                    sampler = SingleGaussian1DSampler()
                    sampler.fit(np.array([min_v, max_v]))
                    val = sampler.sample_with_confidence(n_samples=1, conf_thresh=conf)[0]
                    candidate_vector[f_id] = np.clip(val, min_v, max_v)

                    quality_score = (n_splits + 1.0) / ((span + 1e-6) * (1.0 + path_error))
                    scores_vector[f_id] = quality_score

                tree_candidates.append(candidate_vector)
                tree_scores.append(scores_vector)

            tree_candidates = np.array(tree_candidates)
            tree_scores = np.array(tree_scores)
            combined_sample = np.zeros(self.n_features)

            for j in range(self.n_features):
                best_tree = np.argmax(tree_scores[:, j])
                combined_sample[j] = tree_candidates[best_tree, j]

            gen_samples[i] = combined_sample

        return gen_samples


def create_complete_random_feat_groups(n_groups: int, features_each_group: float, total_features: int) -> List[List[int]]:
    int_array = list(range(0, total_features))
    random.shuffle(int_array)
    groups = []
    for i in range(n_groups):
        groups.append(int_array[:features_each_group])
        random.shuffle(int_array)
    return groups
