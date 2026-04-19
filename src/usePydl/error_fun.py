import numpy as np
import src.chanceCalc.prob_estimator as probEstimator

def gaussian_error(samples):
    def error(tids):
        sub_samples = samples[list(tids)]
        if len(sub_samples) <= 1:
            return np.inf
        features = sub_samples.T
        total_estimated_error = 0.0
        distrs = []
        for feat_array in features:
            distrs.append(probEstimator.gaussian_distr(feat_array))
        total_estimated_error += np.sum(probEstimator.get_error_sample(distrs,features.T))
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