import numpy as np
from pyexpat import features

import CONFIG
from src.data.encoders.encoder_PCA import PCAEncoder
from src.data.encoders.encoder_NN import NNencoder

class GlobalChunkInfo:
    def __init__(self, loader):
        self.loader = loader
        self.global_pca = None
        self.global_nn = None

        self.feats_min_vals = None
        self.feats_max_vals = None
        self.processed_feat_min = None
        self.processed_feat_max = None
        self.feature_importance = None

        self.featureTypes = None
        self.discrete_values = None
        self.total_number_samples = 0

        self.init_all_parameters()

    def init_all_parameters(self):
        self.init_default_vals()

        if CONFIG.ROTATE_DIM:
            self.init_pca()
            self.feature_importance = self.global_pca.get_explained_variance_ratio()

        if CONFIG.REDUCE_FEAT is not None:
            self.init_nn()
            if CONFIG.REDUCE_FEAT is not None:
                self.feature_importance = self.global_nn.get_explained_variance_ratio()





    def init_default_vals(self):
        for i in range(self.loader.n_chunks):
            chunk_samples, _ = self.loader.load_chunk(i)
            chunk_samples = convert_samples_to_num(chunk_samples)
            self.total_number_samples += chunk_samples.shape[0]
            if self.feats_min_vals is None: self.feats_min_vals = np.full(chunk_samples.shape[1], np.inf)
            if self.feats_max_vals is None: self.feats_max_vals = np.full(chunk_samples.shape[1], -np.inf)

            #update min-max values
            for i in range(chunk_samples.shape[1]):
                col = chunk_samples[:, i]
                self.feats_min_vals[i] = min(self.feats_min_vals[i], float(np.min(col)))
                self.feats_max_vals[i] = max(self.feats_max_vals[i], float(np.max(col)))

            #update discrete
            self.update_discrete(chunk_samples)

    def update_discrete(self, samples,discrete_percentile=5):
        if self.discrete_values is None: self.discrete_values = [[] for _ in range(samples.shape[1])]
        if self.featureTypes is None: self.featureTypes = [[] for _ in range(samples.shape[1])]
        for i in range(samples.shape[1]):
            u_vals = list(np.unique(samples[:,i].T))
            u_len = len(u_vals)
            if self.total_number_samples > 0 and u_len <= (self.total_number_samples * discrete_percentile // 100):
                self.discrete_values[i] = list(set(u_vals+self.discrete_values[i]))
                self.featureTypes[i] = 'discrete'
            else:
                self.featureTypes[i] = 'continuous'
                self.discrete_values[i] = []

    def init_pca(self):
        for i in range(self.loader.n_chunks):
            chunked_samples, _ = self.loader.load_chunk(i)
            if CONFIG.APPLY_SCALE: chunked_samples = self.scale_and_standardize_samples(chunked_samples)
            if self.global_pca is None:
                self.global_pca = PCAEncoder(output_dim=min(chunked_samples.shape[1], chunked_samples.shape[0]))
            self.global_pca.partial_fit(chunked_samples)


    def init_nn(self):
        for i in range(self.loader.n_chunks):
            chunked_samples, _ = self.loader.load_chunk(i)
            if CONFIG.APPLY_SCALE: chunked_samples = self.scale_and_standardize_samples(chunked_samples)
            if CONFIG.ROTATE_DIM: chunked_samples = self.global_pca.transform(chunked_samples)

            if CONFIG.REDUCE_FEAT >= chunked_samples.shape[1]:
                CONFIG.REDUCE_FEAT = None
                return
            if CONFIG.USE_NN:
                if self.global_nn is None: self.global_nn = NNencoder(CONFIG.REDUCE_FEAT)
            else:
                if self.global_nn is None: self.global_nn = PCAEncoder(CONFIG.REDUCE_FEAT)
            self.global_nn.partial_fit(chunked_samples)


    def load_chunk_data(self, chunk_id):
        return self.loader.load_chunk(chunk_id)


    def scale_and_standardize_samples(self,samples):
        num_feat = []
        featurs = samples.T
        for i,feat in enumerate(featurs):
            num_feat.append(self.scale_and_standardize(feat, i))
        return np.array(num_feat).T.astype(float)

    def scale_and_standardize(self, raw_feature_data, feat_id):
        num_arr = make_num(raw_feature_data)
        min_val = self.feats_min_vals[feat_id]
        max_val = self.feats_max_vals[feat_id]
        if max_val - min_val == 0:    return np.zeros(len(num_arr))
        return np.array((num_arr - min_val) / (max_val - min_val))


def convert_samples_to_num(samples):
        num_feat = []
        featurs = samples.T
        for i,feat in enumerate(featurs):
            num_feat.append(make_num(feat))
        return np.array(num_feat).T.astype(float)


def make_num(raw_feature_data):
    try:
        num_arr = np.asarray(raw_feature_data, dtype=float)
    except (ValueError, TypeError):
        unique_values = np.unique(raw_feature_data)
        indexes = {val: idx for idx, val in enumerate(unique_values)}
        num_arr = np.array([indexes[val] for val in raw_feature_data]).astype(float)
        #maybe store the original strings? but would never need them (always want it to be numbers)
    return num_arr