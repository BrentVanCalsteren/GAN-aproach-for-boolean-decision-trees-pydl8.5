import numpy as np
from sklearn.mixture import GaussianMixture

def get_distr_for_features(features,distr_fun:str="norm"):
    distr_funs = []
    for feat in features:
        if distr_fun == "norm":
            distr_funs.append(normal_distr(feat))
    return distr_funs

def calc_likelihood_scores_samples(distributions, samples):
    total_log_likelihood = 0.0
    features = samples.T
    for i, distr_fun in enumerate(distributions):
        log_likelihoods = distr_fun.score_samples(features[i].reshape(-1, 1)) #returns log_lik
        total_log_likelihood += np.sum(log_likelihoods)
    return total_log_likelihood



def normal_distr(points):#only works well when the data is clustered symmetric around one point
    """Single gaussian -> bell-curve, symmetric"""
    gm = GaussianMixture(n_components=1, covariance_type='full')
    gm.fit(points.reshape(-1, 1))
    #ddof is Delta Degrees of Freedom: ddof=0 assume intire pop, ddof=1 sample from a larger population, to get an unbiased estimate
    return gm