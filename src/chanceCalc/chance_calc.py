import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde

def calc_chance_distr_features(features,distr_fun:str="norm"):
    distr_funs = []
    for feat in features:
        if distr_fun == "norm":
            distr_funs.append(["norm", normal_distr(feat)])
        elif distr_fun == "gaus_kde":
            distr_funs.append(["gaus_kde", gaussian_dens_distr(feat)])
    return distr_funs

def calc_chance_of_single_sample_ind(distributions,sample):
    #we asume each feature is independent in the sample
    total_prob = 1
    for i in range(distributions):
        fun_name, distr_fun = distributions[i]
        feat = sample[i]
        if distr_fun == "norm":
            prob = distr_fun.cdf(feat)
            total_prob *= prob
        elif distr_fun == "gaus_kde":
            prob = distr_fun.evaluate(feat)
            total_prob *= prob
    return total_prob



def normal_distr(points):#only works well when the data is clustered symmetric around one point
    """Single gaussian -> bell-curve, symmetric"""
    mean = np.mean(points)
    standard_dev = np.std(points, ddof=1)
    #ddof is Delta Degrees of Freedom: ddof=0 assume intire pop, ddof=1 sample from a larger population, to get an unbiased estimate
    return norm(mean,standard_dev)

def gaussian_dens_distr(points):
    # works better for unknown data because it can handle gaps within the data intervals
    """for each point calc standart gaus and sum"""
    kde = gaussian_kde(points)
    return kde