import random
import numpy as np

def create_complete_random_feat_groups(n_groups: int, features_each_group: float, total_features: int):
    int_array = list(range(0, total_features))
    random.shuffle(int_array)
    groups = []
    for i in range(n_groups):
        groups.append(int_array[:features_each_group])
        random.shuffle(int_array)
    return groups


def create_feat_cor_imp_groups(n_groups: int, features_each_group: int,total_features: int, feature_importance: np.ndarray = None, corr_matrix: np.ndarray = None):

    if total_features <= features_each_group:
        return [list(range(total_features)) for _ in range(n_groups)]

    k = min(features_each_group, total_features)

    if feature_importance is not None and len(feature_importance) == total_features:
        imp = np.asarray(feature_importance, dtype=float)
        imp = np.maximum(0.0, imp)
        p = (imp / (np.sum(imp) + 1e-8)) * 0.8 + (0.2 / float(total_features))
        p /= np.sum(p)

    else: p = np.ones(total_features, dtype=float) / float(total_features)

    groups = []
    coverage_counter = np.zeros(total_features, dtype=int)
    all_feats = np.arange(total_features)

    for i in range(n_groups):
        group = []
        least_covered = int(np.argmin(coverage_counter))
        group.append(least_covered)
        coverage_counter[least_covered] += 1

        if corr_matrix is not None and corr_matrix.shape == (total_features, total_features):
            corrs = np.abs(corr_matrix[least_covered])
            corrs[least_covered] = -1.0
            top_corr = np.argsort(corrs)[::-1]
            for idx in top_corr:
                if len(group) >= k: break
                if idx not in group:
                    group.append(int(idx))
                    coverage_counter[idx] += 1

        while len(group) < k:
            rem_p = p.copy()
            rem_p[group] = 0.0
            sum_p = np.sum(rem_p)
            if sum_p > 0:
                rem_p /= sum_p
                choice = int(np.random.choice(total_features, p=rem_p))
            else:
                avail = [f for f in all_feats if f not in group]
                choice = int(np.random.choice(avail))
            group.append(choice)
            coverage_counter[choice] += 1

        groups.append(group)

    return groups
