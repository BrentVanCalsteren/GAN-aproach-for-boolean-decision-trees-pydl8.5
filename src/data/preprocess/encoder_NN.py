import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import CONFIG


def create_encoder_decoder_struct(input_dim, output_dim):

    # For large input dimensions (e.g., 1000 -> 50), use a progressive funnel
    if input_dim > 200 and output_dim < input_dim // 2:
        h1 = min(512, max(output_dim * 4, input_dim // 2))
        h2 = min(256, max(output_dim * 2, h1 // 2))

        encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.2),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.2),
            nn.Linear(h2, output_dim),
        )

        decoder = nn.Sequential(
            nn.Linear(output_dim, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.2),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.2),
            nn.Linear(h1, input_dim)
        )
    else:
        h_dim = max(output_dim * 2, input_dim)
        encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, output_dim),
        )
        decoder = nn.Sequential(
            nn.Linear(output_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, input_dim),
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
        self.samples = None
        self.tensor_samples = None
        self.device = "cpu"
        print(f'torch device = {self.device}')
        self.optimizer = None

    def create_new_nn_module(self, samples: np.ndarray) -> None:
        if samples.ndim > 2:
            print("!this encoder expects a 2D grid as input!")
            return

        input_dim = samples.shape[1]

        encoder, decoder = create_encoder_decoder_struct(input_dim, self.output_dim)
        self.nn_module = Module(encoder, decoder).to(self.device)
        self.optimizer = optim.AdamW(self.nn_module.parameters(), lr=0.003, weight_decay=1e-4)

    def partial_fit(self, chunk_samples: np.ndarray, batch_size: int = CONFIG.CHUNK_SIZE, lr: float = 0.003) -> None:
        if chunk_samples.ndim > 2:
            print("!this encoder expects a 2D grid as input!")
            return

        if self.nn_module is None:
            self.create_new_nn_module(chunk_samples)

        if self.optimizer is None:
            self.optimizer = optim.AdamW(self.nn_module.parameters(), lr=lr, weight_decay=1e-4)

        target_error = getattr(CONFIG, 'NN_AVG_MIN_ERROR', 0.005)
        max_epochs = getattr(CONFIG, 'MAX_NN_EPOCHS', 200)

        loader, _ = create_loader_torch(chunk_samples, batch_size)
        mse_criterion = nn.MSELoss()
        cosine_criterion = nn.CosineSimilarity(dim=1)
        self.nn_module.train()

        epoch = 0
        current_loss = float('inf')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)

        while current_loss > target_error and epoch < max_epochs:
            loss_batch = 0.0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.nn_module(batch_x)

                loss_mse = mse_criterion(outputs, batch_x)
                loss_cos = 1.0 - torch.mean(cosine_criterion(outputs, batch_x))
                loss = loss_mse + 0.1 * loss_cos

                loss.backward()
                self.optimizer.step()
                loss_batch += loss.item()

            current_loss = loss_batch / len(loader)
            epoch += 1
            scheduler.step()

        print(f"Partial fit finished after {epoch} epochs. Final Loss: {current_loss:.6f} (target <= {target_error})")

    def train_module(self, batch_size=64, min_error=0.001):
        if self.nn_module is None or self.samples is None:
            print('Need module and samples')
            return

        print(f"Training nn module until min error < {min_error}")
        loader, self.tensor_samples = create_loader_torch(self.samples, batch_size)

        if self.optimizer is None:
            self.optimizer = optim.AdamW(self.nn_module.parameters(), lr=0.003, weight_decay=1e-4)
        mse_criterion = nn.MSELoss()
        self.nn_module.train()

        epochs = 0
        current_loss = float('inf')

        while current_loss > min_error and epochs < 300:
            loss_batch = 0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.nn_module(batch_x)
                loss = mse_criterion(outputs, batch_x)
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

        return resampled

    def get_explained_variance_ratio(self, samples: np.ndarray = None) -> np.ndarray:
        if self.nn_module is None:
            return np.ones(self.output_dim) / float(self.output_dim)

        if samples is None:
            samples = self.samples

        if samples is not None and samples.size > 0:
            latent = self.transform(samples)
            latent_var = np.var(latent, axis=0)
        else:
            latent_var = np.ones(self.output_dim)

        first_dec_layer = self.nn_module.decoder[0]
        if hasattr(first_dec_layer, 'weight'):
            dec_norms = torch.norm(first_dec_layer.weight, dim=0).cpu().detach().numpy()
        else:
            dec_norms = np.ones(self.output_dim)

        importance = latent_var * dec_norms
        total_imp = np.sum(importance)
        if total_imp > 0:
            return importance / total_imp
        return np.ones(self.output_dim) / float(self.output_dim)

    def get_cumulative_explained_variance(self, samples: np.ndarray = None) -> float:
        return float(np.sum(self.get_explained_variance_ratio(samples)))


def create_loader_torch(samples: np.ndarray, batch_size: int):
    tensor_x = torch.tensor(samples, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, tensor_x