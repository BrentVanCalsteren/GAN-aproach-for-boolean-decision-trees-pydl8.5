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
    def generate_new_samples_for_all_features_of_this_type(cls, n: int, conf_thresh: float,samplers: list) -> np.ndarray:
        #For independent 1D samplers, returns a list of fitted samplers.
        #multi-feat samplers will overrite this
        gen_feats = []
        for sampler in samplers:
            gen_feat_good = np.array([])
            while gen_feat_good.size < n:
                gen_feat = sampler.sorted_samples(n=n)
                scores = sampler.score_feature(gen_feat)
                if gen_feat_good.size > 0:
                    gen_feat_good = np.concatenate((gen_feat_good, gen_feat[scores >= conf_thresh])).flatten()
                else:
                    gen_feat_good = gen_feat[scores >= conf_thresh]
            gen_feats.append(gen_feat_good[:n])
        return np.array(gen_feats).T