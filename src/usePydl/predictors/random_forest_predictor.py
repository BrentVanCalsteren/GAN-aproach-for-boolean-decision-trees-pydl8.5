import numpy as np
from typing import List, Optional
from scipy.stats import truncnorm

import CONFIG
from src.data.data_obj.feature_history import FeatureHistory
from src.usePydl.predictors.greedy_deepening_predictor import build_tree_iteratively
from usePydl.predictors.helpers.tree import Tree, calc_intervals_of_path
from src.data.data_obj.sampels import Samples
from usePydl.predictors.helpers.interval import Intervals
import src.usePydl.predictors.helpers.groups as groups


def _train_single_class_forest(args):
    samples_obj, max_features_per_tree, subsample_ratio, cls = args
    print(f"--- [Worker] Training Dedicated Forest for Class {cls} ---")
    sub_predictor = RandomForestPredictor(
        samples_obj,
        max_features_per_tree=max_features_per_tree,
        subsample_ratio=subsample_ratio,
        train_label_class=cls
    )
    return int(cls), sub_predictor



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
                    total_features=self.n_features,
                    feature_importance=self.samples_obj.chunk_info.feature_importance,
                    corr_matrix=self.samples_obj.chunk_info.class_corrs.get(self.train_label_class, None)
                )
            for group in feature_groups:
                print(f"Tree(chunk:{chunk_id}, feat:{group}) TRAINING")
                feat_hist = FeatureHistory(sub_samples)
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
            self.samples_obj.clear_chunk_cache(chunk_id)

    def gen_new_data(self, n: int = 100, reference_samples: Optional[np.ndarray] = None,
            conf: float = 0.8, attempts_per_tree: int = 5,
            ensemble_attempts: int = 25):

        if reference_samples is None: reference_samples = self.samples_obj.samples

        reference_samples = np.asarray(reference_samples, dtype=float)
        labels = self.samples_obj.labels

        if reference_samples.ndim == 1:
            reference_samples = reference_samples.reshape(1, -1)

        n_ref = reference_samples.shape[0]
        if n_ref == 0:
            return np.zeros((0, self.n_features))

        rng = np.random.default_rng()
        indices_ref = rng.choice(n_ref, size=n, replace=True)

        gen_samples = np.zeros((n, self.n_features))

        #####################################################
        #helpers
        #####################################################
        def central_bounds(lo: float, hi: float, left_closed: bool, right_closed: bool):
            if lo == hi: return lo, hi
            if lo > hi: lo, hi = hi, lo
            span = hi - lo
            margin = np.clip((1 - conf),0,1) * span

            a = lo + margin
            b = hi - margin

            #open boundaries handeled
            if not left_closed: a = max(a, np.nextafter(lo, hi))
            if not right_closed: b = min(b, np.nextafter(hi, lo))
            return float(a), float(b)

        def sample_truncated_gaussian(a: float, b: float, mu: float, std_scale: float = 4.0) -> float:
            if a == b: return float(a)
            if a > b: a, b = b, a
            span = b - a
            sigma = max(1e-6, span / std_scale)
            mu_clamped = np.clip(mu, a, b) if mu is not None else (a + b) / 2.0

            a_norm = (a - mu_clamped) / sigma
            b_norm = (b - mu_clamped) / sigma
            try:
                val = truncnorm.rvs(a_norm, b_norm, loc=mu_clamped, scale=sigma)
                return float(np.clip(val, a, b))
            except Exception:
                val = np.random.normal(loc=mu_clamped, scale=sigma)
                return float(np.clip(val, a, b))

        def sample_from_interval(interval_obj, ref_feat: float = None, global_min: float = 0.0, global_max: float = 1.0) -> float:
            g_span = max(1e-5, global_max - global_min)
            if interval_obj is None or interval_obj.is_empty:
                noise = np.random.normal(0.0, 0.03 * g_span)
                val = (ref_feat if ref_feat is not None else (global_min + global_max) / 2.0) + noise
                return float(np.clip(val, global_min, global_max))

            intervals = interval_obj.get_domain_intervals()
            intervals = [inter for inter in intervals if inter.isvalid_interval]
            if not intervals:
                noise = np.random.normal(0.0, 0.03 * g_span)
                val = (ref_feat if ref_feat is not None else (global_min + global_max) / 2.0) + noise
                return float(np.clip(val, global_min, global_max))

            inter_id = 0
            if ref_feat is not None:
                for f_id, inter in enumerate(intervals):
                    if inter.startpoint <= ref_feat <= inter.endpoint:
                        inter_id = f_id
                        break
                else:
                    dists = [min(abs(ref_feat - iv.startpoint), abs(ref_feat - iv.endpoint)) for iv in intervals]
                    inter_id = int(np.argmin(dists))

            inter = intervals[inter_id]
            a, b = central_bounds(inter.startpoint, inter.endpoint, inter.left_closed, inter.right_closed)

            if a == b:
                noise = np.random.normal(0.0, 0.01 * g_span)
                return float(np.clip(a + noise, global_min, global_max))

            mu = ref_feat if ref_feat is not None else (a + b) / 2.0
            return sample_truncated_gaussian(a, b, mu=mu, std_scale=4.0)

        def sample_candidate_from_interval_dict(interval_d, ref_s):
            cand = np.asarray(ref_s, dtype=float).copy()
            if cand.shape[0] != self.n_features: return None

            global_mins = self.samples_obj.chunk_info.processed_feat_min
            global_maxs = self.samples_obj.chunk_info.processed_feat_max

            for f_id in range(self.n_features):
                interval_obj = interval_d.get(f_id, None)
                g_min = float(global_mins[f_id]) if global_mins is not None else 0.0
                g_max = float(global_maxs[f_id]) if global_maxs is not None else 1.0
                cand[f_id] = sample_from_interval(interval_obj, cand[f_id], global_min=g_min, global_max=g_max)

            return cand

        def compactness_interval(interval) -> float:
            if interval is None or interval.is_empty:
                return 0.0

            total = 0.0
            for inter in interval.get_domain_intervals():
                if not inter.isvalid_interval: continue
                if inter.startpoint == inter.endpoint: total += 1e-12
                else: total += max(0.0, inter.endpoint - inter.startpoint)
            return float(total)

        def count_splits(path_dict) -> int:
            if not path_dict: return 0
            total = 0
            for value in path_dict.vals():
                try: total += len(value[0])
                except Exception: pass
            return int(total)

        def intersect_interval_dicts(interval_dicts, feat_hists):
            if not interval_dicts: return {}
            combined = {}
            global_mins = self.samples_obj.chunk_info.processed_feat_min
            global_maxs = self.samples_obj.chunk_info.processed_feat_max

            for f_id in range(self.n_features):
                base_hist = feat_hists[0]
                g_min = float(global_mins[f_id]) if global_mins is not None else 0.0
                g_max = float(global_maxs[f_id]) if global_maxs is not None else 1.0
                combined_obj = Intervals(f_id, base_hist, g_min, g_max)

                empty_found = False
                for interval_dic in interval_dicts:
                    intr = interval_dic.get(f_id, None)
                    if intr is None: continue
                    if intr.is_empty:
                        empty_found = True
                        break

                    combined_obj.add_constraint_union(intr.get_domain_intervals())
                    if combined_obj.is_empty:
                        empty_found = True
                        break

                # Soft Union Fallback if multi-tree intersection is over-constrained/empty
                if empty_found or combined_obj.is_empty:
                    soft_obj = Intervals(f_id, base_hist, g_min, g_max)
                    soft_intervals = []
                    for interval_dic in interval_dicts:
                        intr = interval_dic.get(f_id, None)
                        if intr and not intr.is_empty:
                            soft_intervals.extend(intr.get_domain_intervals())
                    if soft_intervals:
                        soft_obj.interval_list = soft_intervals
                        combined[f_id] = soft_obj
                    else:
                        combined[f_id] = Intervals(f_id, base_hist, g_min, g_max)
                else:
                    combined[f_id] = combined_obj

            return combined

        def score_candidate(path_dict,interval_dic,path_error: float,rel_prob: float,
                candidate: np.ndarray,ref_sample: np.ndarray) -> float:
            """
            The score prefers:1 more specific paths,
                2 higher leaf rel_prob,
                3 lower path error,
                4 smaller allowed intervals,
                5 samples close to the reference sample."""
            n_splits = float(count_splits(path_dict))
            total_len = 0.0
            for f_id in range(self.n_features):
                total_len += compactness_interval(interval_dic.get(f_id, None))

            avg_len = total_len / float(max(1, self.n_features))
            dist = float(np.linalg.norm(candidate - ref_sample))
            score = (n_splits + 1.0) * float(rel_prob)
            score /= ((1.0 + float(path_error))* (1.0 + avg_len)* (1.0 + dist))
            return float(score)

        def candidate_matches_tree(tree, candidate: np.ndarray, target_leaf_id) -> bool:
            try:
                res = tree.get_leaf_id_for_sample(candidate)
                cand_leaf = int(res[0])
                return cand_leaf == int(target_leaf_id)
            except Exception: return False

        #####################################################

        # Pre-cache interval_dic per leaf_id for each tree (fast + high quality)
        tree_leaf_caches = []
        for t_idx, tree in enumerate(self.ensemble_trees):
            feat_hist_t = self.tree_feat_histories[t_idx]
            leaf_cache = {}
            for path_obj in tree.get_all_paths():
                try:
                    p_dict = path_obj.get("path", {})
                    l_id = path_obj.get("leaf_id", 0)
                    interval_dic = calc_intervals_of_path(p_dict, feat_hist_t)
                    leaf_cache[l_id] = interval_dic
                except Exception:
                    pass
            tree_leaf_caches.append(leaf_cache)

        gen_labels = []
        for i, ref_id in enumerate(indices_ref):
            ref_sample = np.asarray(reference_samples[ref_id], dtype=float)
            ref_label = labels[ref_id] if labels is not None and ref_id < len(labels) else 0
            used_trees = []
            used_hists = []
            interval_dicts = []
            path_dicts = []
            leaf_ids = []
            tree_candidates = []
            tree_scores = []
            tree_valid = []

            for t_idx, tree in enumerate(self.ensemble_trees):
                leaf_cache = tree_leaf_caches[t_idx]
                feat_hist_t = self.tree_feat_histories[t_idx]
                try: res = tree.get_leaf_id_for_sample(ref_sample)
                except Exception: continue
                if len(res) < 2: continue

                leaf_id = res[0]
                path_dict = res[1]
                path_error = float(res[2]) if len(res) > 2 else 0.0
                rel_prob = float(res[3]) if len(res) > 3 else 1.0

                interval_dic = leaf_cache.get(leaf_id, None)
                if interval_dic is None:
                    try: interval_dic = calc_intervals_of_path(path_dict, feat_hist_t)
                    except Exception: continue

                if not interval_dic: continue

                best_cand = None
                best_score = -np.inf
                best_valid = False

                for _ in range(max(1, int(attempts_per_tree))):
                    cand = sample_candidate_from_interval_dict(interval_dic, ref_sample)
                    valid = candidate_matches_tree(tree, cand, leaf_id)
                    score = score_candidate(path_dict, interval_dic, path_error, rel_prob, cand, ref_sample)
                    if valid: score *= 10.0

                    if score > best_score:
                        best_score = score
                        best_cand = cand
                        best_valid = valid

                if best_cand is None:
                    best_cand = ref_sample.copy()
                    best_valid = candidate_matches_tree(tree, best_cand, leaf_id)
                    best_score = score_candidate(path_dict, interval_dic, path_error, rel_prob, best_cand, ref_sample)

                used_trees.append(tree)
                used_hists.append(feat_hist_t)
                interval_dicts.append(interval_dic)
                path_dicts.append(path_dict)
                leaf_ids.append(leaf_id)

                tree_candidates.append(best_cand)
                tree_scores.append(best_score)
                tree_valid.append(best_valid)

            if not interval_dicts:
                gen_samples[i] = ref_sample
                continue

            try:
                combined_intervals = intersect_interval_dicts(interval_dicts, used_hists)
                best_ensemble_candidate = None
                best_ensemble_score = -np.inf
                found_valid_ensemble = False

                for _ in range(max(1, int(ensemble_attempts))):
                    cand = sample_candidate_from_interval_dict(combined_intervals, ref_sample)
                    valid_all = True
                    for tree, leaf_id in zip(used_trees, leaf_ids):
                        if not candidate_matches_tree(tree, cand, leaf_id):
                            valid_all = False
                            break

                    dist = float(np.linalg.norm(cand - ref_sample))
                    ens_score = 1.0 / (1.0 + dist)

                    if valid_all: ens_score *= 100.0

                    if ens_score > best_ensemble_score:
                        best_ensemble_score = ens_score
                        best_ensemble_candidate = cand
                        found_valid_ensemble = valid_all

                    if found_valid_ensemble: break

                if best_ensemble_candidate is not None:
                    if found_valid_ensemble:
                        gen_samples[i] = best_ensemble_candidate
                        continue
            except Exception: pass

            if tree_candidates:
                tree_scores_arr = np.asarray(tree_scores, dtype=float)
                tree_valid_arr = np.asarray(tree_valid, dtype=bool)

                valid_indices = np.where(tree_valid_arr)[0]

                if len(valid_indices) > 0:
                    best_idx = int(valid_indices[np.argmax(tree_scores_arr[valid_indices])])
                    gen_samples[i] = tree_candidates[best_idx]
                else:
                    best_idx = int(np.argmax(tree_scores_arr))
                    gen_samples[i] = tree_candidates[best_idx]
            else:
                gen_samples[i] = ref_sample
            gen_labels.append(ref_label)
        return gen_samples


