import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.robust import scale
from torch.utils.data import DataLoader, TensorDataset
# ==============================================================================
def create_encoder_decoder_struct(input_dim,output_dim):
    sqrt_in = int(math.sqrt(input_dim))
    sqrt_out = int(math.sqrt(output_dim))
    encoder = nn.Sequential(
        # Step 1: Reshape flat input [batch, input_dim] to [batch, channels, height, width]
        nn.Unflatten(1, (1,sqrt_in,sqrt_in)),
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*6*6, output_dim)
    )

    decoder = nn.Sequential(
        nn.Unflatten(1, (1,sqrt_out,sqrt_out)),
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*2*2, input_dim)
    )
    return encoder, decoder

class Module(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        if self.encoder is None or self.decoder is None:
            print('Need to have a encoder and a decoder structure')
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class NN_encoder:
    def __init__(self, output_dim=20):
        self.output_dim = output_dim
        self.nn_module = None
        self.range_samples = (0,1)
        self.samples = None
        self.tensor_samples = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_new_nn_module(self, samples: np.ndarray, epochs=10, batch_size=32) -> np.ndarray:
        if samples.ndim > 2:
            print("!this encoder expects a 2d grid as input!")
            return samples
        if np.max(samples) > 1 or np.min(samples) < 0:
            print("samples are not scaled correctly, rescaling")
            samples, self.range_samples = scale(samples)
        self.samples = samples
        input_dim = samples.shape[1]
        encoder, decoder = create_encoder_decoder_struct(input_dim, self.output_dim)
        module = Module(encoder, decoder)
        self.nn_module = module

    def train_module(self, batch_size=32, min_error=0.001):
        if self.nn_module is None or self.samples is None:
            print('Need module and samples')
            return
        print(f"Training nn module until min error is lower: {min_error}")
        samples = self.samples
        module = self.nn_module
        loader, self.tensor_samples = create_loader_torch(samples, batch_size)
        optimizer = optim.Adam(module.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        module.train()
        total_loss = 1
        iterations = 0
        while total_loss/len(loader) > min_error:
            loss_batch = 0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                outputs = module(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                loss_batch += loss.item()
            total_loss = loss_batch/len(loader)
            iterations+=1
            if (iterations + 1) % 50 == 0:
                print(f"Iteration {iterations}, Loss: {total_loss/len(loader):.4f}")
                
        # Transform the data

    def transform(self, samples) -> np.ndarray:
        if samples.ndim > 2:
            print("!this encoder expects a 2d grid as input!")
            return samples
        self.nn_module.eval()
        with torch.no_grad():
            tensor_samples = self.tensor_samples.to(self.device)
            reduced = self.nn_module.encoder(tensor_samples).cpu().numpy()

        return reduced

    def inverse_transform(self, samples: np.ndarray) -> np.ndarray:
        if self.nn_module is None:
            print('cant inverse scale since there is no module present')
            raise samples
            
        self.nn_module.eval()
        with torch.no_grad():
            tensor_reduced = torch.tensor(samples, dtype=torch.float32).to(self.device)
            resampled = self.nn_module.decoder(tensor_reduced).cpu().numpy()

        resampled_scaled = reverse_scale(resampled,self.range_samples)
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
        return np.zeros(arr.shape)
    return np.array((arr - min_val) / (max_val - min_val)), (min_val, max_val)

def reverse_scale(arr: np.ndarray, range):
    min_val,max_val = range
    if max_val - min_val == 0:
        return np.full_like(arr, min_val)
    return arr * (max_val - min_val) + min_val