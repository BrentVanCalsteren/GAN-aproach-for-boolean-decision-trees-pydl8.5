import numpy as np
import src.chanceCalc.prob_estimator as probEstimator

def prob_norm_error(samples,p):
    def error(tids):
        sub_samples = np.array(samples[list(tids)])
        n = len(sub_samples)
        if n <= 1:
            return 1
        total_prob = 0
        for i in range(len(sub_samples)):
            sample = sub_samples[i]
            other = np.delete(sub_samples, i, axis=0)
            features = other.T
            distributions = probEstimator.get_gaussian_distributions(features)
            sample_prob = probEstimator.calc_normalised_confidence_gaussian(distributions,np.array([sample]))
            if np.min(sample_prob) > p:
                total_prob += np.min(sample_prob)
            else:
                total_prob -= np.min(sample_prob)
        return -total_prob
    return error

##TEST FUNCTION
def prob_norm_error2(samples):
    def error(tids):
        sub_samples = samples[list(tids)]
        if len(sub_samples) <= 1:
            return np.inf
        features = sub_samples.T
        total_estimated_error = 0.0
        for feat_array in features:
            gm = probEstimator.gaussian_distr(feat_array)
            total_estimated_error += probEstimator.get_error(np.array([gm]),np.array([feat_array]))[0]
        #print(total_estimated_error)
        return total_estimated_error
    return error

##########################
#FAST ERROR FUN
#########################

def sim_error(y):
    def error(tids):
        supports = list(tids) # is a list of all counts
        maxindex = np.argmax(supports) #take label that is most promenent
        return sum(supports) - supports[maxindex], maxindex #returns how many missclassified
    return error