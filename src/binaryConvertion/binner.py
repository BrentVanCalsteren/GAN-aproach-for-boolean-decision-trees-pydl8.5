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
    indices = [0] * len(array)
    n_bins = len(clusters)
    for i in range(len(array)):
        converted = False
        closest_val = 1
        for cluster_idx, cluster_values in enumerate(clusters):
            if isinstance(cluster_values, np.float64):
                min_dif = abs(array[i] - cluster_values)
            elif isinstance(cluster_values, list):
                left, right = cluster_values
                min_dif = min(abs(array[i] - left),abs(array[i] - right))
            else:
                raise (ValueError, print(cluster_values))
            if min_dif < closest_val:
                closest_val = min_dif
                indices[i] = cluster_idx
                converted = True
        if not converted:
            raise ValueError(f'Problem: not converted index:{i}')

    onehot_strings = []
    try:
        for idx in indices:
            arr = ['0'] * n_bins
            arr[idx] = '1'
            onehot_strings.append(''.join(arr))
    except IndexError:
        print("something wrong")


    return np.array(onehot_strings)




def bin_convertion_2d(array_2d, max_bins=16) -> Tuple[np.ndarray, np.ndarray, List]:
    """uses bin_convertion function one row at a time for 2d_array"""
    processed_features = []
    bin_length = []
    clusters = []
    for feature_row in array_2d:
        onehot_strings, cluster = bin_convertion(feature_row, max_bins=max_bins)
        processed_features.append(onehot_strings)
        clusters.append(cluster)
        bin_length.append(len(onehot_strings[0]) if onehot_strings[0] else 0)
    return np.array(processed_features), np.array(bin_length), clusters

def flatten_binary_strings(bin_strings):
    combined = ''.join(bin_strings)
    return [int(ch) for ch in combined]


