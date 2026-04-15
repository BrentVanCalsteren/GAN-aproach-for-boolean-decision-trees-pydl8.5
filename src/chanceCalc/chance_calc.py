import numpy as np
from sklearn.mixture import GaussianMixture

def get_distr_for_features(features):
    distr_funs = []
    for feat in features:
        distr_funs.append(normal_distr(feat))
    return distr_funs #[dstr-f1, dstr-f2, dstr-f3,...]

def calc_likelihood_scores_samples(distributions, samples):
    features = samples.T
    #[F1 F1 F1]
    #[F2 F2 F2]
    likes_exp = []
    for i, distr_fun in enumerate(distributions): #[DF1 DF2 ...]
        log_likelihoods = distr_fun.score_samples(features[i,:].reshape(-1, 1)) #returns log_lik
        #returns prob dens vals: pdf(x) = 1 / (σ * √(2π)) * exp( - (x - μ)² / (2σ²) )
        #density of 2.5 as "2.5 times more confident"—it reflects sharper distribution.
        likes_exp.append(np.exp(log_likelihoods))
        # [LF1 LF1 LF1]
        # [LF2 LF2 LF2]
    samples_prob_matrix = np.array(likes_exp).T
    return samples_prob_matrix #[LF1 LF2]
                               #[LF1 LF2]

##TEST FUNCTION
def calc_normalised_confidence(distributions, samples):
    features = samples.T  # (n_features, n_samples)
    log_likelihoods = np.zeros(samples.shape[0])
    log_likelihoods_max = np.zeros(samples.shape[0])

    for i, gm in enumerate(distributions):
        # Log-likelihood for all samples
        ll = gm.score_samples(features[i].reshape(-1, 1))
        log_likelihoods += ll

        # Maximum log-likelihood for this feature (at the mean)
        mu = gm.means_[0, 0]
        ll_max = gm.score_samples([[mu]])[0]
        log_likelihoods_max += ll_max

    # Normalise
    confidence = np.exp(log_likelihoods - log_likelihoods_max)
    return confidence


def normal_distr(points):#only works well when the data is clustered symmetric around one point
    """Single gaussian -> bell-curve, symmetric"""
    gm = GaussianMixture(n_components=1, covariance_type='full')
    gm.fit(points.reshape(-1, 1))
    return gm