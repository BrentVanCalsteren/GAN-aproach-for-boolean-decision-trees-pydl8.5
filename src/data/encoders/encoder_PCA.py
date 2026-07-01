import numpy as np
from sklearn.decomposition import PCA

class PCAEncoder:
    def __init__(self, output_dim=None, information_kept=0.9):
        if output_dim is None:
            self.pca = PCA(n_components=information_kept)
        else:
            self.pca = PCA(n_components=output_dim)

    def transform(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim > 2:
            print('!expects a 2d grid!')
            return samples
        return self.pca.fit_transform(samples)

    def inverse_transform(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim > 2:
            print('!expects a 2d grid!')
            return samples
        return self.pca.inverse_transform(samples)
