import numpy as np
#continue errors
#/////////////////////////////////////
#default predictor error
def predictor_error(predictor, samples: np.ndarray):
    def error(tids):
        sub_samples = samples[list(tids)]
        n_samples, n_features = sub_samples.shape
        if n_features // 2 >= n_samples:
            return np.inf
        features = sub_samples.T
        distrs = predictor.get_distributions(features)
        return np.sum(predictor.get_error_sample(distrs, sub_samples)) / n_samples
    return error


#other errors
def mse_error(samples: np.ndarray):
    def error(tids):
        sub_samples = samples[list(tids)]
        if len(sub_samples) <= 1:
            return 1
        mean = np.mean(sub_samples, axis=0)
        return np.mean(np.sum((sub_samples - mean) ** 2, axis=1))

    return error

def mae_error(samples):
    def error(tids):
        sub_samples = samples[list(tids)]
        if len(sub_samples) <= 1:
            return 1
        pred = np.mean(sub_samples, axis=0)
        return np.mean(np.sum(np.abs(sub_samples - pred), axis=1))
    return error


def huber_error(samples, delta: float = 1.35):
    def error(tids):
        sub_samples = samples[list(tids)]
        pred = np.median(sub_samples, axis=0)
        diff = sub_samples - pred
        abs_diff = np.abs(diff)
        #Huber loss: 0.5 * diff² if |diff| ≤ delta, else delta * (|diff| - 0.5 * delta)
        loss = np.where(abs_diff <= delta, 0.5 * (diff ** 2), delta * (abs_diff - 0.5 * delta))
        return np.mean(np.sum(loss, axis=1))

    return error


#Binned/Discrete Data
def total_entropy_error(samples):
    def error(tids):
        sub_samples = samples[list(tids)]
        features = sub_samples.T
        total_entropy = 0.0
        for feature in features:
            total_entropy+=entropy_error_feature(feature)
    return error

def entropy_error_feature(feature):
    _, counts = np.unique(feature, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))

def total_gini_error(samples):
    def error(tids):
        sub_samples = samples[list(tids)]
        features = sub_samples.T
        total_gini = 0.0
        for feature in features:
            total_gini += entropy_error_feature(feature)
    return error

def gini_error_feat(feature):
    #count class occurrences
    _, counts = np.unique(feature, return_counts=True)
    probs = counts / counts.sum()
    gini = 1.0 - np.sum(probs ** 2)
    return gini


##########################
#FAST ERROR FUN
#########################

def min_sup_error(y):
    def error(tids):
        supports = list(tids) # is a list of all counts
        maxindex = np.argmax(supports) #take label that is most promenent
        return sum(supports) - supports[maxindex], maxindex #returns how many missclassified
    return error