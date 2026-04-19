from typing import Tuple, List, Union

from exceptiongroup import catch

from src.clustering1D.divisive_clustering_1D import DivisiveCluster
import numpy as np
import math

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
    processed_features = []
    clusters = []
    feat_bin_len = []
    for feature_row in array_2d:
        onehot_strings, cluster = bin_convertion(feature_row, max_bins=max_bins)
        processed_features.append(onehot_strings)
        feat_bin_len.append(len(onehot_strings[0]))
        clusters.append(cluster)
    flattend_bin = np.array([flatten_binary_strings(row) for row in np.array(processed_features).T])
    return flattend_bin, feat_bin_len,clusters

def flatten_binary_strings(bin_strings):
    combined = ''.join(bin_strings)
    return [int(ch) for ch in combined]


