from sklearn.neighbors import NearestNeighbors
import numpy as np

def do_all_sanity_checks(real_samples,generated_samples):
    result = data_dim_type_mismatch(real_samples,generated_samples)
    print(f"sanity type mismatch: {result}")
    result = nearest_neighbor_check(real_samples,generated_samples)
    print(f"nearest neighbour avg dist feat: {result}")


def data_dim_type_mismatch(real_samples:np.ndarray, gen_samples:np.ndarray):
    """Score: 0 = no mismatch, 1 = complete mismatch."""
    real_feat = real_samples.T
    gen_feat = gen_samples.T
    if real_feat.shape[0] != gen_feat.shape[0]:
        raise ValueError("Incompatible number of features")
    diffs = 0
    for i, _ in enumerate(real_feat):
        if real_feat[i].dtype != gen_feat[i].dtype:
            diffs += 1
    return diffs / (real_feat.shape[0] + 1)


def nearest_neighbor_check(real_samples:np.ndarray, gen_samples:np.ndarray):
    """Average distance from real samples to nearest synthetic neighbor."""
    estimator = NearestNeighbors(n_neighbors=1).fit(gen_samples)
    dist, _ = estimator.kneighbors(real_samples, return_distance=True)
    return np.mean(dist)

def check_discrete_or_continue_data(real_samples:np.ndarray, gen_samples:np.ndarray):
    #TODO: still need to make gen samples back discrete when the original vals are discrete
    pass