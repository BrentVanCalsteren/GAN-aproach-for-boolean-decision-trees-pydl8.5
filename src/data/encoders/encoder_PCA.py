import numpy as np
from sklearn.decomposition import PCA

class PCAEncoder:
    def __init__(self, target_variance=0.98, compentents = None):
        if compentents is None:
            self.pca = PCA(n_components=target_variance)
        else:
            self.pca = PCA(n_components=compentents)
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 2:
            # Flatten multi-dimensional data (e.g. images)
            X = X.reshape(X.shape[0], -1)
        
        reduced_X = self.pca.fit_transform(X)
        self.is_fitted = True
        print(f"HigherDimEncoder: Reduced from {X.shape[1]} to {reduced_X.shape[1]} components.")
        return reduced_X

    def inverse_transform(self, X_reduced: np.ndarray, original_shape=None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before inverse_transform is called.")
        
        X_reconstructed = self.pca.inverse_transform(X_reduced)
        
        if original_shape is not None:
            X_reconstructed = X_reconstructed.reshape(original_shape)
            
        return X_reconstructed
