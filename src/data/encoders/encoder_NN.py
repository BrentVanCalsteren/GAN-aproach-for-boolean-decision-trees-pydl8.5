import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NN_untrained_Encoder:
    def __init__(self, target_dim=20):
        self.target_dim = target_dim
        self.W = None
        self.W_pinv = None
        self.original_shape = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.original_shape = X.shape[1:]
        X_flat = X.reshape(X.shape[0], -1)
        
        input_dim = X_flat.shape[1]
        
        # Create random weights (untrained neural network layer)
        # Scaled using standard Xavier/Glorot initialization logic
        self.W = np.random.randn(input_dim, self.target_dim) / np.sqrt(input_dim)
        
        # Calculate pseudo-inverse for exact mathematical reverse-feeding (reconstruction)
        self.W_pinv = np.linalg.pinv(self.W)
        
        # Forward pass (Reduce)
        reduced = X_flat @ self.W
        print(f"UntrainedRandomNNReducer: Reduced from {input_dim} to {self.target_dim}")
        return reduced

    def inverse_transform(self, X_reduced: np.ndarray, original_shape=None) -> np.ndarray:
        if self.W_pinv is None:
            raise RuntimeError("Must fit_transform before inverse_transform")
            
        # Reverse feed (Reconstruct original features)
        reconstructed = X_reduced @ self.W_pinv
        
        shape = original_shape if original_shape is not None else (-1, *self.original_shape)
        return reconstructed.reshape(shape)

# ==============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TrainedAutoencoderReducer:
    """
    Uses a trained Neural Network (Autoencoder) to reduce features.
    This is the modern Deep Learning way to compress and decompress data.
    It trains briefly to learn the optimal non-linear reduction.
    """
    def __init__(self, target_dim=20, epochs=10, batch_size=32):
        self.target_dim = target_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.original_shape = None
        self.x_min = None
        self.x_max = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.original_shape = X.shape[1:]
        X_flat = X.reshape(X.shape[0], -1)
        
        # Normalize between 0 and 1 for the Neural Net
        self.x_min = X_flat.min(axis=0)
        self.x_max = X_flat.max(axis=0)
        
        range_vals = self.x_max - self.x_min
        range_vals[range_vals == 0] = 1.0 # Prevent division by zero
        
        X_norm = (X_flat - self.x_min) / range_vals
        
        input_dim = X_norm.shape[1]
        self.autoencoder = Autoencoder(input_dim, self.target_dim).to(self.device)
        
        tensor_x = torch.tensor(X_norm, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"Training Autoencoder for {self.epochs} epochs on {self.device}...")
        self.autoencoder.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                outputs = self.autoencoder(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
                
        # Transform the data
        self.autoencoder.eval()
        with torch.no_grad():
            tensor_x = tensor_x.to(self.device)
            reduced = self.autoencoder.encoder(tensor_x).cpu().numpy()
            
        return reduced

    def inverse_transform(self, X_reduced: np.ndarray, original_shape=None) -> np.ndarray:
        if self.autoencoder is None:
            raise RuntimeError("Must fit_transform before inverse_transform")
            
        self.autoencoder.eval()
        with torch.no_grad():
            tensor_reduced = torch.tensor(X_reduced, dtype=torch.float32).to(self.device)
            reconstructed_norm = self.autoencoder.decoder(tensor_reduced).cpu().numpy()
            
        # Denormalize
        range_vals = self.x_max - self.x_min
        range_vals[range_vals == 0] = 1.0
        reconstructed = reconstructed_norm * range_vals + self.x_min
        
        shape = original_shape if original_shape is not None else (-1, *self.original_shape)
        return reconstructed.reshape(shape)
