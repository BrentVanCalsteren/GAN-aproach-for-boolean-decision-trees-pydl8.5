import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde

def get_distr_for_features(features,distr_fun:str="norm"):
    distr_funs = []
    for feat in features:
        if distr_fun == "norm":
            distr_funs.append(["norm", normal_distr(feat)])
        elif distr_fun == "gaus_kde":
            distr_funs.append(["gaus_kde", gaussian_dens_distr(feat)])
    return distr_funs

def calc_chance_for_single_sample(all_feat_distrs, sample):
    #we asume each feature is independent in the sample
    total_prob = 0
    for i in range(len(all_feat_distrs)):
        fun_name, distr_fun = all_feat_distrs[i]
        feat = sample[i]
        if fun_name == "norm":
            prob = distr_fun.cdf(feat)
            total_prob += prob
        elif fun_name == "gaus_kde":
            prob = distr_fun.evaluate(feat)
            total_prob += prob
    if len(all_feat_distrs) > 1: total_prob /= len(all_feat_distrs)
    return total_prob



def normal_distr(points):#only works well when the data is clustered symmetric around one point
    """Single gaussian -> bell-curve, symmetric"""
    mean = np.mean(points)
    standard_dev = np.std(points, ddof=0)
    #ddof is Delta Degrees of Freedom: ddof=0 assume intire pop, ddof=1 sample from a larger population, to get an unbiased estimate
    return norm(mean,standard_dev)

def gaussian_dens_distr(points):
    # works better for unknown data because it can handle gaps within the data intervals
    """for each point calc standart gaus and sum"""
    kde = gaussian_kde(points)
    return kde