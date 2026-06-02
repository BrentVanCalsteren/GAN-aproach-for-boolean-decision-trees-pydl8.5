import numpy as np

class Sampler:
    def fit(self, points):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def score_feature(self, feature):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def score_avg(self, feature):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def sample(self, n_samples=1):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def sorted_samples(self, n: int) -> np.ndarray:
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def get_error(self, feature):
        return 1 - self.score_feature(feature)

    @classmethod
    def fit_all_features_of_this_type(cls, features: np.ndarray) -> list:
        #For independent 1D samplers, returns a list of fitted samplers.
        #multi-feat samplers will overrite this
        samplers = []
        for feat in features:
            sampler = cls()
            sampler.fit(feat)
            samplers.append(sampler)
        return samplers

    @classmethod
    def generate_new_samples_for_all_features_of_this_type(cls, indices,gen_feats_matrix, conf_thresh: float, samplers: list):
        bundeld_feats = []
        n = gen_feats_matrix.shape[1]
        for sampler in samplers:
            single_feat = np.array([])
            while len(single_feat) < n:
                gen_feat = sampler.sorted_samples(n=n)
                scores = sampler.score_feature(gen_feat)
                gen_feat_good = gen_feat[scores >= conf_thresh]
                if gen_feat_good.size > 0:
                    if single_feat.size > 0:
                        single_feat = np.concatenate((single_feat, gen_feat_good))
                    else:
                        single_feat = gen_feat_good
            bundeld_feats.append(single_feat[:n])
        gen_feats_matrix[indices] = np.array(bundeld_feats)