from typing import Dict, List

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
    def generate_new_samples_for_all_features_of_this_type(cls, indices, gen_feats_matrix, conf_thresh: float,samplers: list, intervals_list: List = None):
        bundled_feats = []
        if gen_feats_matrix.ndim == 1:
            raise ValueError('if you want to use this method you gen_matrix needs to be 2d')
        n = gen_feats_matrix.shape[1]
        for i, sampler in enumerate(samplers):
            single_feat = np.array([])
            attempts = 0
            min_v, max_v = None, None
            if intervals_list is not None and i < len(intervals_list):
                min_v, max_v = intervals_list[i].get_complete_domain()

            while len(single_feat) < n and attempts < 100:
                gen_feat = sampler.sorted_samples(n=n)
                if min_v is not None and max_v is not None:
                    gen_feat = np.clip(gen_feat, min_v, max_v)
                scores = sampler.score_feature(gen_feat)
                valid_mask = scores >= conf_thresh
                gen_feat_good = gen_feat[valid_mask]
                if gen_feat_good.size > 0:
                    if single_feat.size > 0:
                        single_feat = np.concatenate((single_feat, gen_feat_good))
                    else:
                        single_feat = gen_feat_good
                if attempts == 50:
                    print('difficult finding exemples')
                attempts += 1

            if min_v is not None and max_v is not None and single_feat.size > 0:
                single_feat = np.clip(single_feat, min_v, max_v)

            bundled_feats.append(single_feat[:n].flatten())
        gen_feats_matrix[indices] = np.array(bundled_feats)