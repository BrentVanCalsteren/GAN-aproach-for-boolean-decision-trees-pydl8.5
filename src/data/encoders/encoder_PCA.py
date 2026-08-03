import numpy as np
from sklearn.decomposition import IncrementalPCA


class PCAEncoder:
    def __init__(self, output_dim=None, batch_size=None):
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.pca = IncrementalPCA(n_components=output_dim, batch_size=batch_size)

    def partial_fit(self, samples: np.ndarray) -> None:
        if samples.ndim > 2:
            print('!expects a 2d grid!')
            return
        self.pca.partial_fit(samples)

    def fit(self, samples: np.ndarray) -> None:
        self.partial_fit(samples)

    def transform(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim > 2:
            print('!expects a 2d grid!')
            return samples
        return self.pca.transform(samples)

    def inverse_transform(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim > 2:
            print('!expects a 2d grid!')
            return samples
        return self.pca.inverse_transform(samples)

    def get_explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    def get_cumulative_explained_variance(self) -> float:
        return float(np.sum(self.pca.explained_variance_ratio_))
