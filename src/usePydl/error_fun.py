import numpy as np
from scipy.spatial.distance import pdist
#continue errors


class CombinedMSEIntervalError:
    def __init__(self, samples: np.ndarray, feature_weights=None, mse_weight: float = 0.6, interval_weight: float = 0.4):
        self.samples = samples
        self.mse_weight = mse_weight
        self.interval_weight = interval_weight
        self.interval_err = IntervalSizesError(samples, feature_weights)
        self.mse_err = MSEError(samples, feature_weights)
        #normalization
        all_ids = np.arange(samples.shape[0], dtype=np.intc)
        self.root_interval = max(1e-5, float(self.interval_err(all_ids)))
        self.root_mse = max(1e-5, float(self.mse_err(all_ids)))
    def __call__(self, tids):
        indices = np.fromiter(tids, dtype=np.intc)
        if len(indices) < 2:
            return 10e6
        norm_i = self.interval_err(indices) / self.root_interval
        norm_m = self.mse_err(indices) / self.root_mse
        return (self.interval_weight * norm_i) + (self.mse_weight * norm_m)

#dense bounding boxes
class IntervalSizesError:
    def __init__(self, samples, feature_weights=None):
        self.samples = samples
        self.feature_weights = np.asarray(feature_weights, dtype=float) if feature_weights is not None else None

    def __call__(self, tids):
        sub_samples = self.samples[list(tids)]
        if sub_samples.shape[0] < 2:
            return 10e6

        max_per_row = sub_samples.max(axis=0)
        min_per_row = sub_samples.min(axis=0)
        diff = max_per_row - min_per_row
        if self.feature_weights is not None:
            return np.sum(diff * self.feature_weights)
        return np.sum(diff)


#finds spherical clusters
class MSEError:
    def __init__(self, samples: np.ndarray, feature_weights=None):
        self.samples = samples
        self.feature_weights = np.asarray(feature_weights, dtype=float) if feature_weights is not None else None

    def __call__(self, tids):
        indices = np.fromiter(tids, dtype=np.intc)
        sub_samples = self.samples[indices]
        if sub_samples.shape[0] < 2:
            return 10e6
        mean = np.mean(sub_samples, axis=0)
        diff_sq = (sub_samples - mean) ** 2
        if self.feature_weights is not None:
            return np.mean(np.sum(diff_sq * self.feature_weights, axis=1))
        return np.mean(np.sum(diff_sq, axis=1))

#Mean Absolute Error
class MAEError:
    def __init__(self, samples: np.ndarray):
        self.samples = samples

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