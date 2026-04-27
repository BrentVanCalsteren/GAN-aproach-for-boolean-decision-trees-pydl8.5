#statistical ways to check difference in data

import numpy as np
from scipy import stats, spatial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

#TODO: further research
def univariate_ks_distance(X_real, X_gen):
    distances = []
    for j in range(X_real.shape[1]):
        d, _ = stats.ks_2samp(X_real[:, j], X_gen[:, j])
        distances.append(d)
    return np.mean(distances), distances

def univariate_wasserstein_distance(X_real, X_gen):
    distances = []
    for j in range(X_real.shape[1]):
        d = stats.wasserstein_distance(X_real[:, j], X_gen[:, j])
        distances.append(d)
    return np.mean(distances), distances

def correlation_matrix_diff(X_real, X_gen, method='pearson'):
    if method == 'pearson':
        corr_real = np.corrcoef(X_real, rowvar=False)
        corr_gen  = np.corrcoef(X_gen,  rowvar=False)
    else:  # spearman
        corr_real, _ = stats.spearmanr(X_real)
        corr_gen, _  = stats.spearmanr(X_gen)
    diff = corr_real - corr_gen
    # Normalize by maximum possible difference (2 * number of off-diagonal elements)
    n = corr_real.shape[0]
    max_diff = np.sqrt(2 * n * (n-1))  # maximum Frobenius norm for correlation matrices
    return np.linalg.norm(diff, 'fro') / max_diff


def propensity_mse(X_real, X_gen, clf=None, cv=5):
    n_real = X_real.shape[0]
    n_gen = X_gen.shape[0]
    # Combine and create labels: 1 = real, 0 = generated
    X_combined = np.vstack([X_real, X_gen])
    y_combined = np.hstack([np.ones(n_real), np.zeros(n_gen)])

    # Default classifier
    if clf is None:
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)

    # Cross-validated predictions (probability of being real)
    preds = cross_val_predict(clf, X_combined, y_combined, cv=cv, method='predict_proba')
    p_real = preds[:, 1]  # probability of class 'real'

    # pMSE = mean( (p - 0.5)^2 )
    pmse = np.mean((p_real - 0.5) ** 2)

    # Normalized pMSE: divide by the theoretical maximum (0.25) for uninformative classifier
    normalized_pmse = pmse / 0.25

    return pmse, normalized_pmse


def mmd_rbf(X, Y, gamma=None):
    if gamma is None:
        # Median heuristic for sigma
        distances = spatial.distance.pdist(np.vstack([X, Y]))
        sigma = np.median(distances) if len(distances) > 0 else 1.0
        gamma = 1.0 / (2 * sigma ** 2)

    K_XX = np.exp(-gamma * spatial.distance.squareform(spatial.distance.pdist(X) ** 2))
    K_YY = np.exp(-gamma * spatial.distance.squareform(spatial.distance.pdist(Y) ** 2))
    XY = spatial.distance.cdist(X, Y, 'euclidean')
    K_XY = np.exp(-gamma * XY ** 2)

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return max(mmd, 0)  # numerical rounding


def evaluate_similarity(X_real, X_gen):
    ks_mean, ks_per_feat = univariate_ks_distance(X_real, X_gen)
    ws_mean, ws_per_feat = univariate_wasserstein_distance(X_real, X_gen)
    corr_diff = correlation_matrix_diff(X_real, X_gen, method='pearson')
    pmse, norm_pmse = propensity_mse(X_real, X_gen)

    print(f"Avg KS distance:          {ks_mean:.4f}")
    print(f"Avg Wasserstein distance: {ws_mean:.4f}")
    print(f"Correlation matrix diff:  {corr_diff:.4f}")
    print(f"pMSE (normalized):        {norm_pmse:.4f}  (1 = indistinguishable)")

    return {
        "ks": ks_mean,
        "wasserstein": ws_mean,
        "corr_diff": corr_diff,
        "pmse_norm": norm_pmse
    }

real = np.random.normal(0, 1, (1000, 5))
gen  = np.random.normal(0, 1, (1000, 5))  # identical distribution → perfect scores
gen_bad = np.random.uniform(-2, 2, (1000, 5))  # different → poor scores

print("Perfect generation:")
evaluate_similarity(real, gen)
print("\nSuboptimal generation:")
evaluate_similarity(real, gen_bad)