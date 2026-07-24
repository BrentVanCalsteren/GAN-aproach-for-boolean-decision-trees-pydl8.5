import numpy as np
from scipy.spatial.distance import pdist
#continue errors

#dense bounding boxes
class IntervalSizesError:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.good_error = 0.1*samples.shape[1]

    def __call__(self, tids):
        sub_samples = self.samples[list(tids)]
        if sub_samples.shape[0] < 2:
            return 10e6

        max_per_row = sub_samples.max(axis=0)
        min_per_row = sub_samples.min(axis=0)
        diff = max_per_row - min_per_row
        return np.sum(diff)


#finds spherical clusters
class MSEError:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.good_error = 0.02 * samples.shape[1]

    def __call__(self, tids):
        indices = np.fromiter(tids, dtype=np.intc)
        sub_samples = self.samples[indices]
        if sub_samples.shape[0] < 2:
            return 10e6
        mean = np.mean(sub_samples, axis=0)
        return np.mean(np.sum((sub_samples - mean) ** 2, axis=1))

#Mean Absolute Error
class MAEError:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.good_error = 0.1 * samples.shape[1]

    def __call__(self, tids):
        sub_samples = self.samples[list(tids)]
        if sub_samples.shape[0] < 2:
            return 10e6
        median = np.median(sub_samples, axis=0)
        return np.mean(np.sum(np.abs(sub_samples - median), axis=1))


#Ensuring tight, strictly bounded clusters
class DiameterError:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.good_error = 0.10 * np.sqrt(samples.shape[1])

    def __call__(self, tids):
        indices = np.fromiter(tids, dtype=np.intc)
        sub_samples = self.samples[indices]
        if sub_samples.shape[0] < 2:
            return 10e6
        distances = pdist(sub_samples, metric='euclidean')
        if np.max(distances) < self.good_error: return 0
        else: return np.max(distances)

##########################
#FAST ERROR FUN
#########################

def min_sup_error(y):
    def error(tids):
        supports = list(tids) # is a list of all counts
        maxindex = np.argmax(supports) #take label that is most promenent
        return sum(supports) - supports[maxindex], maxindex #returns how many missclassified
    return error