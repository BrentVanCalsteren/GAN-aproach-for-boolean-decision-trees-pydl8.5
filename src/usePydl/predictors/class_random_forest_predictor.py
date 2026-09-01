import numpy as np
from typing import List, Dict

from src.data.data_obj.feature_history import FeatureHistory
from src.usePydl.predictors.greedy_deepening_predictor import build_tree_iteratively
from src.data.data_obj.sampels import Samples
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.usePydl.predictors.random_forest_predictor import RandomForestPredictor


def _train_single_class_chunk_trees(args):
    samples_obj, max_features_per_tree, subsample_ratio, cls, chunk_id = args
    samples_obj.load_chunk(chunk_id)
    chunk_samples = samples_obj.samples
    if cls is not None and samples_obj.labels is not None and len(samples_obj.labels) > 0:
        chunk_samples = chunk_samples[(samples_obj.labels.flatten() == cls)]

    n_total = chunk_samples.shape[0]
    if n_total == 0:
        return int(cls), [], []

    n_sub = int(np.ceil(n_total * subsample_ratio))
    sub_indices = np.random.choice(n_total, size=n_sub, replace=True)
    sub_samples = chunk_samples[sub_indices]

    n_features = samples_obj.samples.shape[1]
    feature_factor = int(np.ceil(np.sqrt(n_features)))
    features_each_tree = int(min(max_features_per_tree, feature_factor + feature_factor / 2))
    trees_each_chunk = max(4, int(np.ceil((n_features / max(1, features_each_tree)) * 3)))

    corr_m = None
    if samples_obj.chunk_info is not None and hasattr(samples_obj.chunk_info, 'class_corrs'):
        corr_m = samples_obj.chunk_info.class_corrs.get(cls, None)

    feature_groups = groups.create_feat_cor_imp_groups(
        n_groups=trees_each_chunk,
        features_each_group=features_each_tree,
        total_features=n_features,
        feature_importance=samples_obj.chunk_info.feature_importance if samples_obj.chunk_info else None,
        corr_matrix=corr_m
    )

    chunk_trees = []
    chunk_hists = []
    for group in feature_groups:
        feat_hist = FeatureHistory(sub_samples, samples_obj.chunk_info)
        weights = feat_hist.get_feature_weights(mode='uniform', focus_on=group)
        feat_hist.creat_splits(weight_of_each_feature=weights)
        pred = build_tree_iteratively(feat_hist, weights)
        complete_tree = feat_hist.get_complete_tree()
        if complete_tree is None or complete_tree.tree is None:
            complete_tree = pred.tree

        complete_tree.remove_sample_ids_from_leafs()
        feat_hist.reduce_memory()
        chunk_trees.append(complete_tree)
        chunk_hists.append(feat_hist)

    return int(cls), chunk_trees, chunk_hists


class ClassRandomForest:
    def __init__(self, samples_obj: Samples, max_features_per_tree: int = 5, subsample_ratio: float = 0.5):
        self.samples_obj = samples_obj
        self.labels = samples_obj.labels.flatten() if samples_obj.labels is not None else np.zeros(samples_obj.samples.shape[0], dtype=int)
        self.unique_classes = np.unique(self.labels)
        self.class_forests: Dict[int, RandomForestPredictor] = {}
        self.class_reference_samples: Dict[int, List[np.ndarray]] = {cls: [] for cls in self.unique_classes}
        self.n_features = samples_obj.samples.shape[1]

        print(f"Init Class-Conditioned Forest for {len(self.unique_classes)} classes: {self.unique_classes}")
        self.build_class_forests(max_features_per_tree, subsample_ratio)

    def build_class_forests(self, max_features_per_tree: int, subsample_ratio: float):
        n_chunks = self.samples_obj.loader.n_chunks
        max_workers = min(len(self.unique_classes), 5)
        print(f"Parallelizing {len(self.unique_classes)} Class Sub-Forests over {n_chunks} chunk(s) (1 chunk cached in RAM at a time)...")

        # Instantiate empty sub-forests for each class
        for cls in self.unique_classes:
            self.class_forests[cls] = RandomForestPredictor(
                self.samples_obj,
                max_features_per_tree=max_features_per_tree,
                subsample_ratio=subsample_ratio,
                train_label_class=cls,
                auto_build=False
            )

        # Chunk-major outer loop: load chunk k -> collect class reference samples -> train sub-forests -> evict chunk k
        for chunk_id in range(n_chunks):
            print(f"=== [Chunk {chunk_id + 1}/{n_chunks}] Loading chunk data into RAM cache ===")
            self.samples_obj.load_chunk(chunk_id)

            chunk_labels = self.samples_obj.labels.flatten() if self.samples_obj.labels is not None else np.zeros(self.samples_obj.samples.shape[0], dtype=int)
            for cls in self.unique_classes:
                cls_s = self.samples_obj.samples[(chunk_labels == cls)]
                if len(cls_s) > 0 and sum(len(x) for x in self.class_reference_samples[cls]) < 200:
                    self.class_reference_samples[cls].append(cls_s)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_train_single_class_chunk_trees,
                                    (self.samples_obj, max_features_per_tree, subsample_ratio, cls, chunk_id)): cls
                    for cls in self.unique_classes
                }
                for future in as_completed(futures):
                    cls, new_trees, new_hists = future.result()
                    self.class_forests[cls].ensemble_trees.extend(new_trees)
                    self.class_forests[cls].tree_feat_histories.extend(new_hists)

            # Evict chunk_id from RAM cache immediately after all classes finish training on it
            self.samples_obj.clear_chunk_cache(chunk_id)
            print(f"=== [Chunk {chunk_id + 1}/{n_chunks}] Evicted from RAM cache ===")

        # Stack collected per-class reference samples
        for cls in self.unique_classes:
            if self.class_reference_samples[cls]:
                self.class_reference_samples[cls] = np.vstack(self.class_reference_samples[cls])
            else:
                self.class_reference_samples[cls] = np.zeros((0, self.n_features))

    def gen_new_data(self, target_class=None, n: int = 100, conf: float = 0.8):
        if target_class is not None:
            if target_class not in self.class_forests:
                raise ValueError(f"Class {target_class} not found in trained class forests.")
            cls_ref_samples = self.class_reference_samples.get(target_class, np.zeros((0, self.n_features)))
            forest = self.class_forests[target_class]
            samples = forest.gen_new_data(n=n, reference_samples=cls_ref_samples, conf=conf)
            labels = np.full(n, target_class, dtype=int)
            return samples, labels

        num_classes = self.unique_classes.size
        if num_classes == 0:
            return np.zeros((0, self.n_features)), np.zeros(0, dtype=int)

        base_n = n // num_classes
        remainder = n % num_classes

        gen_samples_list = []
        gen_labels_list = []

        for i, cls in enumerate(self.unique_classes):
            current_n = base_n + (1 if i < remainder else 0)
            if current_n == 0:
                continue

            cls_ref_samples = self.class_reference_samples.get(cls, np.zeros((0, self.n_features)))
            forest = self.class_forests[cls]

            print(f'Generating {current_n} new data samples for class {cls}')
            samples = forest.gen_new_data(n=current_n, reference_samples=cls_ref_samples, conf=conf)
            gen_samples_list.append(samples)
            gen_labels_list.append(np.full(current_n, cls, dtype=int))

        if not gen_samples_list:
            return np.zeros((0, self.n_features)), np.zeros(0, dtype=int)

        gen_samples = np.concatenate(gen_samples_list, axis=0)
        gen_labels = np.concatenate(gen_labels_list, axis=0)
        return gen_samples, gen_labels