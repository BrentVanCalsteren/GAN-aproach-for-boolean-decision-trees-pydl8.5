import numpy as np


class Uniform_distr:
    def __init__(self):
        self.min = None
        self.max = None
        self.n_features = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features = X.shape[1]
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)


    def sample(self, n_samples=1):
        if self.min is None:
            raise RuntimeError("Model must be fitted before sampling.")

        return np.random.uniform(
            low=self.min,
            high=self.max,
            size=(n_samples, self.n_features)
        )