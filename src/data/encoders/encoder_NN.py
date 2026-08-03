import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def create_encoder_decoder_struct(input_dim, output_dim):
    hidden_dim = input_dim * 2

    encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Sigmoid()
    )

    decoder = nn.Sequential(
        nn.Linear(output_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
        nn.Sigmoid()
    )

    return encoder, decoder


class Module(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        if self.encoder is None or self.decoder is None:
            raise ValueError('Need to have an encoder and a decoder structure')
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class NNencoder:
    def __init__(self, output_dim=20):
        self.output_dim = output_dim
        self.nn_module = None
        self.range_samples = (0, 1)
        self.samples = None
        self.tensor_samples = None
        self.device = torch.device("cpu")
        self.optimizer = None

    def create_new_nn_module(self, samples: np.ndarray) -> None:
        if samples.ndim > 2:
            print("!this encoder expects a 2D grid as input!")
            return

        if np.max(samples) > 1 or np.min(samples) < 0:
            print("Samples are not scaled correctly, rescaling...")
            samples, self.range_samples = scale(samples)

        self.samples = samples
        input_dim = samples.shape[1]

        encoder, decoder = create_encoder_decoder_struct(input_dim, self.output_dim)
        self.nn_module = Module(encoder, decoder).to(self.device)
        self.optimizer = optim.Adam(self.nn_module.parameters(), lr=0.001)

    def partial_fit(self, chunk_samples: np.ndarray, epochs: int = 5, batch_size: int = 32, lr: float = 0.001) -> None:
        if chunk_samples.ndim > 2:
            print("!this encoder expects a 2D grid as input!")
            return

        if np.max(chunk_samples) > 1 or np.min(chunk_samples) < 0:
            chunk_samples, self.range_samples = scale(chunk_samples)

        if self.nn_module is None:
            self.create_new_nn_module(chunk_samples)

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.nn_module.parameters(), lr=lr)

        loader, _ = create_loader_torch(chunk_samples, batch_size)
        criterion = nn.MSELoss()
        self.nn_module.train()

        for epoch in range(epochs):
            loss_batch = 0.0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.nn_module(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                self.optimizer.step()
                loss_batch += loss.item()

    def train_module(self, batch_size=32, min_error=0.001):
        if self.nn_module is None or self.samples is None:
            print('Need module and samples')
            return

        print(f"Training nn module until min error < {min_error}")
        loader, self.tensor_samples = create_loader_torch(self.samples, batch_size)

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.nn_module.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.nn_module.train()

        epochs = 0
        current_loss = float('inf')

        while current_loss > min_error and epochs < 500:
            loss_batch = 0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.nn_module(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                self.optimizer.step()
                loss_batch += loss.item()

            current_loss = loss_batch / len(loader)
            epochs += 1

            if epochs % 50 == 0:
                print(f"Epoch {epochs}, Loss: {current_loss:.6f}")

    def transform(self, samples: np.ndarray) -> np.ndarray:
        if self.nn_module is None:
            raise ValueError('Module not initialized.')
        if samples.ndim > 2:
            print("! This encoder expects a 2D grid as input !")
            return samples

        if np.max(samples) > 1 or np.min(samples) < 0:
            samples = (samples - self.range_samples[0]) / (self.range_samples[1] - self.range_samples[0] + 1e-8)

        self.nn_module.eval()
        with torch.no_grad():
            tensor_input = torch.tensor(samples, dtype=torch.float32).to(self.device)
            reduced = self.nn_module.encoder(tensor_input).cpu().numpy()

        return reduced

    def inverse_transform(self, samples: np.ndarray) -> np.ndarray:
        if self.nn_module is None:
            raise ValueError('Cannot inverse scale since there is no module present.')

        self.nn_module.eval()
        with torch.no_grad():
            tensor_reduced = torch.tensor(samples, dtype=torch.float32).to(self.device)
            resampled = self.nn_module.decoder(tensor_reduced).cpu().numpy()

        resampled_scaled = reverse_scale(resampled, self.range_samples)
        return resampled_scaled


def create_loader_torch(samples: np.ndarray, batch_size: int):
    tensor_x = torch.tensor(samples, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, tensor_x


def scale(arr: np.ndarray):
    min_val = arr.min()
    max_val = arr.max()

    if max_val - min_val == 0:
        return np.zeros(arr.shape), (min_val, max_val)

    return (arr - min_val) / (max_val - min_val), (min_val, max_val)


def reverse_scale(arr: np.ndarray, val_range: tuple):
    min_val, max_val = val_range
    if max_val - min_val == 0:
        return np.full_like(arr, min_val)
    return arr * (max_val - min_val) + min_val