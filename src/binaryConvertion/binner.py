from typing import Tuple, List
from src.clustering1D.divisive_clustering_1D import DivisiveCluster
import numpy as np

def bin_convertion(n_array, max_bins=16) -> Tuple[np.ndarray, List]:
    """will convert a string/number array into a 1-hot string array"""
    array = np.array(n_array)
    unique_values = np.unique(array)
    clusters = []

    if len(unique_values) > max_bins:
        cluster = DivisiveCluster(max_depth=max_bins, min_cluster_size=1)
        print(f"Clustering array (unique: {len(unique_values)}) with number clusters of {max_bins}")
        cluster.fit(array)
        clusters = cluster.get_clusters()

    else:
        clusters = np.unique(n_array)

    return gen_one_hot_string(array, clusters), clusters

def gen_one_hot_string(array, clusters):
    #clusters is 2d array [[lb,rb], [lb,rb], [lb,rb],...]
    #it can happen that a value in array falls between cluster (when generating data)
    #every value should be mapped to its closest cluster
    array = np.asarray(array)
    n_bins = len(clusters)
    indices = np.zeros(len(array), dtype=int)

    for i, val in enumerate(array):
        best_dist = np.inf
        best_idx = -1

        for idx, cluster in enumerate(clusters):
            if np.isscalar(cluster):
                dist = abs(val - cluster)
            elif isinstance(cluster, (list, tuple, np.ndarray)) and len(cluster) == 2:
                left, right = cluster
                dist = min(abs(val - left), abs(val - right))
            else:
                raise TypeError(f"Unsupported cluster format: {cluster}")

            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx == -1:
            raise ValueError(f"Could not assign value {val} at index {i} to any cluster.")
        indices[i] = best_idx

    onehot_strings = []
    for idx in indices:
        arr = ['0'] * n_bins
        arr[idx] = '1'
        onehot_strings.append(''.join(arr))
    return np.array(onehot_strings)




def bin_convertion_2d(array_2d, max_bins=16) -> Tuple[np.ndarray, List,List]:
    """uses bin_convertion function one row at a time for 2d_array"""
    processed_features = [] #is bin strings
    clusters = []
    feat_bin_len = []
    for feature_row in array_2d:
        onehot_strings, cluster = bin_convertion(feature_row, max_bins=max_bins)
        processed_features.append(onehot_strings)
        feat_bin_len.append(len(onehot_strings[0]))
        clusters.append(cluster)
    compact_features(processed_features)
    flattend_bin = np.array([flatten_binary_strings(row) for row in np.array(processed_features).T])
    return flattend_bin, feat_bin_len,clusters

def compact_features(features_bin_2D):
    counts = get_count_difs(features_bin_2D)
    for i in range(len(counts)):
        for j in range(len(counts[i])):
            if i != j:
                if counts[i][j] < len(features_bin_2D[0])//10:
                    print(f"possible feat dependence:{i,j}")
    for i, feature in enumerate(features_bin_2D):
        if len(feature[0]) == 2:
            features_bin_2D[i] = [f[0] for f in feature]
            print(f'compacted feat of len 2: {i}')


def get_count_difs(features_bin_2D):
    np_array = np.array(features_bin_2D)
    count_matrix = np.zeros(np_array.shape)
    for i, feat in enumerate(np_array):
        unique_values, counts = np.unique(feat, return_counts=True)
        sorted = counts[counts.argsort()]
        count_matrix[i,0:len(counts)] = sorted
    count_difs =  {}
    for i, count in enumerate(count_matrix):
        count_difs[i] = [np.abs(np.array(count_c-count)).sum() for count_c in count_matrix.copy()]
    return count_difs



def flatten_binary_strings(bin_strings):
    combined = ''.join(bin_strings)
    return [int(ch) for ch in combined]


