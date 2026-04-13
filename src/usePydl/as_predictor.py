import random

import numpy as np
from pydl85 import DL85Predictor
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
import helper

complete_x,missing_x,dl_x,scaled_x,bin_length_x,clusters = helper.prep_data_for_pydl_no_sep('bank')
print(scaled_x)

#let' try to make a first Discriminator
"""
def error(tids):
    X_subset = scaled_x[list(tids), :]
    centroid = np.mean(X_subset, axis=0).reshape(1, -1)
    distances = euclidean_distances(X_subset, centroid)
    return float(np.sum(distances))

def leaf_value(tids):
    x_original =  scaled_x[list(tids)]
    centroid = np.mean(x_original, axis=0)
    dists = euclidean_distances(x_original, centroid.reshape(1, -1)).flatten()
    max_dist = np.max(dists) if len(dists) > 0 else 0.0
    return {'centroid': centroid, 'max_dist': max_dist}
    
def confidence(x_new, leaf_info):
    d = np.linalg.norm(x_new - leaf_info['centroid'])
    md = leaf_info['max_dist']
    return max(0.0, 1.0 - d/md) if md > 0 else 1.0
"""
# ── Error function ────────────────────────────────────────────────────────────
# Within-cluster sum of squares (WCSS) — the canonical additive clustering
# criterion. Additive per-sample, so DL8.5's branch-and-bound pruning stays
# valid. Vectorised: no Python loop over tids.
def error(tids):
    X = scaled_x[list(tids)]          # (n, d)
    mu = X.mean(axis=0)               # (d,)
    return float(np.sum((X - mu) ** 2))                        # mean → variance


# ── Leaf value ────────────────────────────────────────────────────────────────
# Store a per-feature Gaussian N(mu_j, sigma_j) for every feature j.
# This is a proper generative model of what "real data in this leaf" looks like.
# scipy.stats.norm is used at score time; here we just store parameters.
def leaf_value(tids):
    X = scaled_x[list(tids)]          # (n, d)
    mu  = X.mean(axis=0)              # (d,)
    std = X.std(axis=0)               # (d,)
    # Floor std so no feature ever collapses to a spike.
    # 0.01 is ~1 % of a [0,1]-scaled feature — tight but not degenerate.
    std = np.maximum(std, 0.01)
    return {'mu': mu, 'std': std, 'n': len(tids)}

# ── Confidence ────────────────────────────────────────────────────────────────
# Product of per-feature Gaussian likelihoods, normalised to [0, 1].
# We work in log-space for numerical stability, then exponentiate.
#
# Raw log-likelihood varies with n_features and std magnitude, so we
# normalise by the *maximum possible* log-likelihood (the score the centroid
# itself would get) to get a value in (0, 1].
def confidence(x_new, leaf_info):
    mu = leaf_info['mu']
    std = leaf_info['std']
    d = len(mu)

    # Average log-likelihood per feature
    log_px = np.sum(norm.logpdf(x_new, loc=mu, scale=std)) / d
    log_pmax = np.sum(norm.logpdf(mu, loc=mu, scale=std)) / d

    return float(np.exp(log_px - log_pmax))


disc = DL85Predictor(
    max_depth=5,
    min_sup=5,
    error_function=error,
    leaf_value_function=leaf_value,
    time_limit=30)

disc.fit(dl_x)

leaf_predictions = disc.predict(dl_x[:10])
test_scores = np.array([confidence(scaled_x[i],x) for i,x in enumerate(disc.predict(dl_x))])
print(f"avg scores real data:{test_scores.mean()}")

#---------------------------------------------------
#as generator -> starting from completely random inputs
#-----------------------------------------------------
#first generating with just de Discriminator
def sample_from_discriminator(disc, n_samples, rng=np.random):
    leafs = disc.predict(dl_x)

    samples = []
    for _ in range(n_samples):
        leaf = rng.choice(leafs)
        params = leaf
        mu = params['mu']
        std = params['std']
        new_x = rng.normal(mu, std)
        new_x = np.clip(new_x, 0, 1)
        samples.append(new_x)
    return np.array(samples)


# Generate 1000 new samples
X_synthetic = sample_from_discriminator(disc, 1000)
print(f'new samples from the leafs of discriminator {X_synthetic} Avg score:'
      f'{np.array([confidence(scaled_x[i],x) for i,x in enumerate(disc.predict(helper.convert_num_specific_bin_length(X_synthetic.T,clusters)))]).mean()}')
"""
n_samples = 100
#creat random numerical noise
_, n_features = scaled_x.shape
z = np.random.random(size=(n_samples, n_features))
z_bin = helper.convert_num_specific_bin_length(z.T,clusters)

#test  replicate original data
original_data = scaled_x

def error_for_building_tree(tids):
    a = np.array(z_bin[list(tids)])
    pred_scores = np.array([confidence(scaled_x[i], x) for i, x in enumerate(disc.predict(a))])
    return float(pred_scores.mean())

def leaf_val_gen(tids):
    a = z[list(tids)]
    return a

# --- 4. Train the generator ---
generator = DL85Predictor(
    max_depth=5,
    min_sup=1,
    error_function=error_for_building_tree,
    leaf_value_function=leaf_val_gen,
    time_limit=300
)

generator.fit(z_bin)
"""




