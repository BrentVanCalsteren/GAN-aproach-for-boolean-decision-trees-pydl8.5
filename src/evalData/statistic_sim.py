import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chisquare
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier

N_BINS = 10

def get_frequency(real_feat:np.ndarray, gen_feat:np.ndarray):
    """Helper to compute aligned histograms for real and synthetic data."""
    freqs = {}
    for i,_ in enumerate(real_feat):
        # Determine if column is numeric
        bins = min(N_BINS, len(np.unique(real_feat[i])))
        real_binned = pd.cut(real_feat[i], bins=bins)
        gen_binned = pd.cut(gen_feat[i], bins=bins)

        real_counts = real_binned.value_counts(normalize=True)
        synth_counts = gen_binned.value_counts(normalize=True)

        # Align
        all_labels = real_counts.index.union(synth_counts.index)
        real_counts = real_counts.reindex(all_labels, fill_value=0)
        synth_counts = synth_counts.reindex(all_labels, fill_value=0)

        freqs[i] = (real_counts.values, synth_counts.values)
    return freqs

def jensenshannon_distance(real_samples:np.ndarray, gen_samples:np.ndarray):
    """Average Jensen-Shannon distance (0 = identical)."""
    real_feat = real_samples.T
    gen_feat = gen_samples.T
    freqs = get_frequency(real_feat, gen_feat)
    res = []
    for i,_ in enumerate(real_feat):
        gt_freq, synth_freq = freqs[i]
        js_dist = jensenshannon(gt_freq, synth_freq)
        if np.isnan(js_dist): js_dist = 0
        res.append(js_dist)
    return np.mean(res)

def chi_squared_test(real_samples:np.ndarray, gen_samples:np.ndarray):
    """Average p-value of chi-squared test (higher = more similar)."""
    real_feat = real_samples.T
    gen_feat = gen_samples.T
    freqs = get_frequency(real_feat, gen_feat)
    res = []
    for i,_ in enumerate(real_feat):
        gt_freq, synth_freq = freqs[i]
        try:
            _, pvalue = chisquare(gt_freq + 1, synth_freq + 1) # Adding 1 to avoid zeros
            if np.isnan(pvalue): pvalue = 0
        except:
            pvalue = 0
        res.append(pvalue)
    return np.mean(res)

def kolmogorov_smirnov_test(real_samples:np.ndarray, gen_samples:np.ndarray):
    """Average (1 - KS statistic) per column (higher = more similar)."""
    real_feat = real_samples.T
    gen_feat = gen_samples.T
    res = []
    for i,_ in enumerate(real_feat):
        statistic, _ = ks_2samp(real_feat[i], gen_feat[i])
        res.append(1 - statistic)
    return np.mean(res)

def inv_kl_divergence(real_samples:np.ndarray, gen_samples:np.ndarray):
    """Average inverse KL divergence (1 = identical, 0 = different)."""
    real_feat = real_samples.T
    gen_feat = gen_samples.T
    freqs = get_frequency(real_feat, gen_feat)
    res = []
    for i,_ in enumerate(real_feat):
        gt_freq, synth_freq = freqs[i]
        kl_div = np.sum(synth_freq * np.log((synth_freq + 1e-9) / (gt_freq + 1e-9)))
        res.append(1 / (1 + kl_div))
    return np.mean(res)

def max_mean_discrepancy(real_samples:np.ndarray, gen_samples:np.ndarray, kernel='rbf'):
    """MMD using RBF kernel (0 = identical distributions)."""
    if kernel == 'rbf':
        gamma = 1.0
        XX = metrics.pairwise.rbf_kernel(real_samples, real_samples, gamma)
        YY = metrics.pairwise.rbf_kernel(gen_samples, gen_samples, gamma)
        XY = metrics.pairwise.rbf_kernel(real_samples, gen_samples, gamma)
        score = XX.mean() + YY.mean() - 2 * XY.mean()
    else: # linear
        delta = real_samples.mean(axis=0) - gen_samples.mean(axis=0)
        score = delta.dot(delta.T)
    return float(score)


def _wasserstein_1d(u, v):
    """Compute 1D Wasserstein distance between two arrays."""
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    all_vals = np.concatenate([u_sorted, v_sorted])
    all_vals.sort()
    # CDFs approximated by linear interpolation
    u_cdf = np.searchsorted(u_sorted, all_vals, side='right') / len(u)
    v_cdf = np.searchsorted(v_sorted, all_vals, side='right') / len(v)
    # Integrate |CDF1 - CDF2|
    delta = np.diff(all_vals)
    # Midpoint differences
    cdf_diff = np.abs(u_cdf[:-1] - v_cdf[:-1])
    return np.sum(cdf_diff * delta)

def wasserstein_distance(real:np.ndarray, gen:np.ndarray,method: str = 'sliced',
    n_projections: int = 50,seed: int = None) -> float:

    if method == 'featurewise':
        d = 0.0
        for j in range(real.shape[1]):
            d += _wasserstein_1d(real[:, j], gen[:, j])
        return d / real.shape[1]

    elif method == 'sliced':
        d = real.shape[1]
        rng = np.random.RandomState(seed)
        dist_sum = 0.0
        for _ in range(n_projections):
            if d == 1:
                direction = np.ones(1)
            else:
                direction = rng.randn(d)
                direction /= np.linalg.norm(direction)
            proj_real = real @ direction #matrix calc lol
            proj_gen  = gen @ direction
            dist_sum += _wasserstein_1d(proj_real, proj_gen)
        return dist_sum / n_projections

    else:
        raise ValueError(f"Unknown method: {method}")

def feature_corr_diff(real_samples:np.ndarray, gen_samples:np.ndarray):
    """Normalized Frobenius norm of correlation matrix difference."""
    corr_real = np.corrcoef(real_samples, rowvar=False)
    corr_synth = np.corrcoef(gen_samples, rowvar=False)
    diff = corr_real - corr_synth
    n = corr_real.shape[0]
    max_diff = np.sqrt(2 * n * (n-1))
    return np.linalg.norm(diff, 'fro') / max_diff

###################################
### function/disctribution detection
###################################"

def eval_with_model(self, model, real_samples:np.ndarray, gen_samples:np.ndarray, **model_args):
    labels_real = np.zeros(real_samples.shape[0])
    labels_gen = np.ones(gen_samples.shape[0])

    data = np.concatenate([real_samples, gen_samples])
    labels = np.concatenate([labels_real, labels_gen])

    skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
    auc_scores = []
    for train_idx, test_idx in skf.split(data, labels):
        train_data, test_data = data[train_idx], data[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        clf = model(**model_args).fit(train_data, train_labels)
        test_pred = clf.predict_proba(test_data)[:, 1]
        score = roc_auc_score(test_labels, test_pred)
        auc_scores.append(score)
    return np.mean(auc_scores)


def detection_linear(real_samples:np.ndarray, gen_samples:np.ndarray):
    """AUC ROC score (0 = indistinguishable)."""
    return eval_with_model(LogisticRegression, real_samples, gen_samples,
                                            solver='lbfgs', max_iter=1000)

def detection_mlp(real_samples:np.ndarray, gen_samples:np.ndarray):
    """AUC ROC score (0 = indistinguishable)."""
    return eval_with_model(MLPClassifier, real_samples, gen_samples,
                                            hidden_layer_sizes=(100,), max_iter=500)

def detection_xgb(real_samples:np.ndarray, gen_samples:np.ndarray):
    """AUC ROC score (0 = indistinguishable)."""
    return eval_with_model(XGBClassifier, real_samples, gen_samples)

def detection_gmm(real_samples:np.ndarray, gen_samples:np.ndarray):
    """Log-likelihood-based detection score (0 = indistinguishable)."""
    gmm_real = GaussianMixture(n_components=min(10, real_samples.shape[0] // 10))
    gmm_gen = GaussianMixture(n_components=min(10, gen_samples.shape[0] // 10))

    # Fit models
    gmm_real.fit(real_samples)
    gmm_gen.fit(gen_samples)

    # Score samples
    real_score_real = gmm_real.score_samples(real_samples)
    real_score_synth = gmm_gen.score_samples(gen_samples)
    # Higher score means the model fits better; we want them to be similar
    acc = np.mean(real_score_real > real_score_synth)
    return 1 - abs(0.5 - acc) * 2  # Normalize to [0, 1] where 0 is best