import numpy as np
from src.data.encoders.encoder_NN import NN_encoder
from src.data.encoders.encoder_PCA import PCAEncoder

def load_encoder(samples: np.ndarray, type='nn', output_dim=20):
    if type == 'nn':
        encoder = NN_encoder(output_dim)
        encoder.create_new_nn_module(samples)
        encoder.train_module()
        return encoder
    elif type == 'pca':
        encoder = PCAEncoder(output_dim)
        return encoder
    else:
        print("type not supported")
        return None

