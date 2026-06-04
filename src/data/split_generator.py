import numpy as np
from src.data.divisive_clustering_1D import DivisiveCluster


def get_best_splits(feature_array, n_splits, is_discrete):
    best_splits = {}
    if is_discrete:
        uniques, counts = np.unique(feature_array, return_counts=True)
        top_uniques = uniques[np.argsort(-counts)][:n_splits]
        for val in top_uniques:
            best_splits[f"even_{val}"] = (feature_array == val).astype(int)
        return best_splits

    #binning percentiles:
    percentiles = np.linspace(5, 95, min(n_splits * 2, 20))
    thresholds = np.unique(np.percentile(feature_array, percentiles))

    #Clustering:
    cluster_thresholds = []
    cluster = DivisiveCluster()
    cluster.max_depth = n_splits
    cluster.fit(feature_array)
    for i in range(n_splits):
        intervals = cluster.get_clusters_at_depth(i)
        for interval in intervals:
            cluster_thresholds.extend([interval[0], interval[1]])

    thresholds = np.unique(np.concatenate((thresholds, cluster_thresholds)))

    candidates = []
    # Candidate generation
    for t in thresholds:
        candidates.append(('bigger', t, (feature_array > t)))
        candidates.append(('smaller', t, (feature_array < t)))

    # Intervals
    for i in range(len(thresholds) - 1):
        for j in range(i + 1, min(i + 5, len(thresholds))):
            t1, t2 = thresholds[i], thresholds[j]
            candidates.append(('interval', (t1, t2), (feature_array >= t1) & (feature_array <= t2)))

    # O(N) Scoring based on range (like user's original logic)
    scored_candidates = []
    total = len(feature_array)
    for c_type, t_val, mask in candidates:
        n_left = np.sum(~mask)
        n_right = np.sum(mask)
        if n_left < 2 or n_right < 2:
            continue

        left_feat = feature_array[~mask]
        right_feat = feature_array[mask]
        err_left = np.max(left_feat) - np.min(left_feat)
        err_right = np.max(right_feat) - np.min(right_feat)
        score = (n_left / total) * err_left + (n_right / total) * err_right
        scored_candidates.append((score, c_type, t_val, mask))

    scored_candidates.sort(key=lambda x: x[0])

    seen_masks = []
    for score, c_type, t_val, mask in scored_candidates:
        if len(best_splits) >= n_splits:
            break

        is_dup = False
        for seen in seen_masks:
            if np.array_equal(mask, seen) or np.array_equal(mask, ~seen):
                is_dup = True
                break

        if not is_dup:
            seen_masks.append(mask)
            if c_type == 'interval':
                best_splits[f"interval_{t_val[0]}_{t_val[1]}"] = mask.astype(int)
            else:
                best_splits[f"{c_type}_{t_val}"] = mask.astype(int)

    return best_splits
