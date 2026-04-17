import numpy as np
from sklearn.mixture import GaussianMixture




##################################################################
#ESTIMATING WITH GAUSSIAN
#########################
def gaussian_distr(feature_array):
    #only works well when the data is clustered symmetric around one point
    """Single gaussian -> bell-curve, symmetric"""
    gm = GaussianMixture(n_components=1, covariance_type='full')
    gm.fit(feature_array.reshape(-1, 1))
    return gm

def get_gaussian_distributions(features):
    distr_funs = []
    for feat in features:
        distr_funs.append(gaussian_distr(feat))
    return distr_funs #[dstr-f1, dstr-f2, dstr-f3,...]

def calc_normalised_confidence_gaussian(distributions, samples):
    features = samples.T  # (n_features, n_samples)
    log_likelihoods = np.zeros(samples.shape[0])
    log_likelihoods_max = np.zeros(samples.shape[0]) #array of feature length
    for i, gm in enumerate(distributions):
        # Log-likelihood for all samples
        log_like = gm.score_samples(features[i].reshape(-1, 1)) #[ll1 ll2 ll3 ...]
        #returns prob dens vals: pdf(x) = 1 / (σ * √(2π)) * exp( - (x - μ)² / (2σ²) )
        #density of 2.5 as "2.5 times more confident"—it reflects sharper distribution.
        log_likelihoods += log_like

        mu = gm.means_[0, 0]#get max log-likelihood (at the mean) for normalizing to value [0,1]
        ll_max = gm.score_samples([[mu]])[0]
        log_likelihoods_max += ll_max #is 1 value but np will map over entire array

    # Normalise
    confidence = np.exp(log_likelihoods - log_likelihoods_max)
    return confidence

def get_error(distributions, samples):
    error = 1 - calc_normalised_confidence_gaussian(distributions, samples)
    return error

##################################################################
#SAME 3 FUNCTIONS FOR OTHER DISTRIBUTION TYPES
#########################

#todo