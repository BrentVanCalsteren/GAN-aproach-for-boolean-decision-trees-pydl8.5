from typing import NamedTuple
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.linalg import qr
import numpy as np


class FeatureStruct(NamedTuple):
    val: float
    feat_index: int


class feat_dependency():
    pass



def get_correlation_matrix(X, threshold=0.95):
    corr_matrix = np.corrcoef(X.T)
    np.fill_diagonal(corr_matrix, 0)  # ignore self-correlation
    correlated_features = np.where(np.abs(corr_matrix) > threshold)
    return list(zip(correlated_features[0], correlated_features[1]))

def get_mutual_info_regression(X, threshold=0.5):
    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Calculate mutual information between features i and j
            mi = mutual_info_regression(X[:, [i]], X[:, j])[0]
            mi_matrix[i, j] = mi_matrix[j, i] = mi

    dependent_pairs = np.where(mi_matrix > threshold)
    return mi_matrix, list(zip(dependent_pairs[0], dependent_pairs[1]))


def get_multi_correlation_matrix(X, threshold=5):
    # Add constant column
    X_with_const = np.column_stack([np.ones(X.shape[0]), X])

    vif_values = []
    for i in range(1, X_with_const.shape[1]):  # skip constant
        vif = variance_inflation_factor(X_with_const, i)
        vif_values.append(vif)

    high_vif_features = np.where(np.array(vif_values) > threshold)[0]
    return vif_values, high_vif_features


def get_linear_dependencies(X, tolerance=1e-10):
    n_features = X.shape[1]
    _, r, perm = qr(X, pivoting=True, mode='r')

    # Find rank
    rank = np.sum(np.abs(np.diag(r)) > tolerance)
    dependent_indices = []
    if rank < n_features:
        # Features in perm[rank:] are linearly dependent
        dependent_indices = perm[rank:]

    return rank, dependent_indices
