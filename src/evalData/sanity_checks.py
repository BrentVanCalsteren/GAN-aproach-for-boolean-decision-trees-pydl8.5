from sklearn.neighbors import NearestNeighbors
import numpy as np

def data_mismatch(self, real_df, synth_df):
    """Score: 0 = no mismatch, 1 = complete mismatch."""
    if real_df.shape[1] != synth_df.shape[1]:
        raise ValueError("Incompatible number of features")
    diffs = 0
    for col in real_df.columns:
        if real_df[col].dtype != synth_df[col].dtype:
            diffs += 1
    return diffs / (len(real_df.columns) + 1)  # Following synthcity's logic


def common_rows_proportion(self, real_df, synth_df):
    """Score: 0 = no common rows, 1 = all real rows leaked."""
    real_hashes = real_df.apply(lambda x: hash(tuple(x)), axis=1)
    synth_hashes = synth_df.apply(lambda x: hash(tuple(x)), axis=1)
    common = len(set(real_hashes).intersection(set(synth_hashes)))
    return common / len(real_df)


def nearest_syn_neighbor_distance(self, real_array, synth_array):
    """Average distance from real samples to nearest synthetic neighbor."""
    estimator = NearestNeighbors(n_neighbors=1).fit(synth_array)
    dist, _ = estimator.kneighbors(real_array, return_distance=True)
    return np.mean(dist)