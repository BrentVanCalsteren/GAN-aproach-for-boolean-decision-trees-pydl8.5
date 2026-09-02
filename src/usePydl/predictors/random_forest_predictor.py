import numpy as np
from typing import List, Optional
from scipy.stats import truncnorm

import CONFIG
from src.data.data_obj.feature_history import FeatureHistory
from src.usePydl.predictors.greedy_deepening_predictor import build_tree_iteratively
from usePydl.predictors.helpers.tree import Tree
from src.data.data_obj.sampels import Samples
from src.samplers.single_gaussian import SingleGaussian1DSampler
from usePydl.predictors.helpers.interval import Interval, add_constraint_union
import src.usePydl.predictors.helpers.groups as groups




class RandomForestPredictor:
    def __init__(self, samples_obj: Samples, max_features_per_tree = 5, subsample_ratio: float = 0.5, train_label_class=None, auto_build: bool = True):
        self.ensemble_trees: List[Tree] = []
        self.tree_feat_histories: List[FeatureHistory] = []
        self.samples_obj = samples_obj
        self.chunk_info = CONFIG.GLOBAL_CHUNK_INFO
        self.n_chunks = samples_obj.loader.n_chunks
        self.n_features = self.samples_obj.samples.shape[1]
        self.features_each_tree = self.n_features
        self.trees_each_chunk = 1
        self.subsample_ratio = subsample_ratio
        self.train_label_class = train_label_class

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
        if auto_build:
            self.build_ensemble()


    def build_ensemble(self):
        for chunk_id in range(self.n_chunks):
            self.samples_obj.load_chunk(chunk_id)
            chunk_samples = self.samples_obj.samples
            if self.train_label_class is not None:
                chunk_samples = chunk_samples[(self.samples_obj.labels.flatten() == self.train_label_class)]
            n_total = chunk_samples.shape[0]
            n_sub = int(np.ceil(n_total * self.subsample_ratio))
            sub_indices = np.random.choice(n_total, size=n_sub, replace=True)
            sub_samples = chunk_samples[sub_indices]
            feature_groups = groups.create_complete_random_feat_groups(n_groups=self.trees_each_chunk,
                                                                       features_each_group=self.features_each_tree,
                                                                       total_features=self.n_features)
            if self.train_label_class is not None:
                feature_groups = groups.create_feat_cor_imp_groups(
                    n_groups=self.trees_each_chunk,
                    features_each_group=self.features_each_tree,
                    total_features=self.n_features)
            for group in feature_groups:
                print(f"Tree(chunk:{chunk_id}, feat:{group}) TRAINING")
                feat_hist = FeatureHistory(sub_samples)
                weights = feat_hist.get_feature_weights(mode='uniform', focus_on=group)
                feat_hist.creat_splits(weight_of_each_feature=weights)
                __predictor = build_tree_iteratively(feat_hist, weights)
                complete_tree = feat_hist.get_complete_tree()
                complete_tree.remove_sample_ids_from_leafs()
                feat_hist.reduce_memory()
                self.ensemble_trees.append(complete_tree)
                self.tree_feat_histories.append(feat_hist)
                print(f"Tree(chunk:{chunk_id}, feat:{group}) Trained")
            self.samples_obj.clear_chunk_cache(chunk_id)

    def gen_new_data_guided(self, n = 100, reference_samples:np.ndarray = None, conf = 0.8, attempts_per_tree = 5):

        if reference_samples is None:
            self.samples_obj.load_chunk(0)
            reference_samples = self.samples_obj.samples

        n_ref = reference_samples.shape[0]
        indices_ref = np.random.default_rng().choice(n_ref, size=n, replace=True)
        gen_samples = np.zeros((n, self.n_features))
        for gen_id,ref_id in enumerate(indices_ref):
            for t in self.ensemble_trees:
                p = t.let_sample_treverse_tree(reference_samples[ref_id])
                splits = p['splits']
                directions = p['directions']
                intervals = []
                for j, d in enumerate(directions):
                    split_obj = splits[j]
                    if d == 'L':
                        intervals.append(split_obj.left_interval())
                    else: intervals.append(split_obj.right_interval())

                intervals_each_feature = {}
                for feat in range(reference_samples.shape[1]):
                    combined_inter = [Interval(self.chunk_info.processed_feat_min[feat], self.chunk_info.processed_feat_max[feat],'closed')]
                    for inter in intervals:
                        inter = inter.get(feat, None)
                        if inter is not None:
                            combined_inter = add_constraint_union(combined_inter, inter)

                    for inter in combined_inter:
                        if inter.contains_value(reference_samples[ref_id,feat]): gen_samples[gen_id,feat] = SingleGaussian1DSampler.sample_from_interval(interval=inter)[0]
        return gen_samples

def score_path(path, intervals_each_feature, path_error, rel_prob):

    def count_splits(path_dict) -> int:
        if not path_dict: return 0
        total = 0
        for value in path_dict.vals():
            try:
                total += len(value[0])
            except Exception:
                pass
        return int(total)

    n_splits = float(len(path['directions']))
    total_len = 0.0
    for feat_id in range(self.n_features):
        total_len += intervals_each_feature[feat_id].get_total_lenght()

    score = (n_splits + 1.0) * float(rel_prob)
    score /= ((1.0 + float(path_error)) * (1.0 + total_len))
    return float(score)


