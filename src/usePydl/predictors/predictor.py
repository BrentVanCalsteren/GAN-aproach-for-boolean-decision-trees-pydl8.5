from typing import List

import numpy as np
from pydl85 import DL85Predictor
import re
from src.usePydl.leaf import get_leafs
from src.usePydl.error_fun import IntervalSizesError, MSEError, DiameterError
from src.usePydl.leaf import ReturnIDSandPROB
from src.samplers.load_samplers import get_sampler_class
from src.samplers.multinomial import MultinomialSampler
from src.samplers.uniform import UniformSampler
from src.samplers.single_gaussian import SingleGaussian1DSampler
from src.samplers.multivariate_gaussian import MultivariateGaussianSampler
from typing import List, Dict, Any

from usePydl.predictors.tree import Tree

COMBINE_FEAT = False


class Predictor:
    n_samples = None

    def __init__(self, splits_obj, samples, max_depth, min_sup, time, n_samples=None):
        print('starting dl predictor...')
        if n_samples is not None:
            self.n_samples = n_samples
        else:
            self.n_samples = samples.shape[0]

        error = IntervalSizesError(samples)
        self.error = error.good_error
        self.dl_predictor = DL85Predictor(error_function=error,
                                          leaf_value_function=ReturnIDSandPROB(self.n_samples),
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)

        self.dl_predictor.fit(splits_obj.get_splits())
        self.tree = Tree(tree=self.dl_predictor.tree_,split_obj=splits_obj)


    def predict(self, samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def get_tree_dict(self):
        return self.tree.tree

    def gen_new_data(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        def inbetween(interval: List[float], val: float) -> bool:
            return interval[0] <= val <= interval[1]

        leafs = self.tree.get_leafs()
        interval_path_dic = self.tree.get_intervals_each_path()
        samples = self.tree.split_obj.sample_obj.samples
        feat_info = self.tree.split_obj.sample_obj.get_feature_info()

        probs_each_path = np.array([leaf['value']['rel_prob'] for leaf in leafs])
        prob_sum = np.sum(probs_each_path)
        if prob_sum > 0:
            probs_each_path /= prob_sum
        else:
            probs_each_path = np.ones(len(probs_each_path)) / len(probs_each_path)
        disc_feat_ids = [feat_inf[0] is not None for feat_inf in feat_info]
        cont_feat_ids = [feat_inf[0] is None for feat_inf in feat_info]
        samples_disc = samples[:, disc_feat_ids]
        samples_cont = samples[:, cont_feat_ids]
        all_new_samples = np.array([])
        indx_list_n = np.random.choice(len(leafs), n, p=probs_each_path)
        indx, counts = np.unique(indx_list_n, return_counts=True)

        for idx, count in zip(indx, counts):
            gen_feat_matrix = np.zeros((len(feat_info), count))
            intervals_each_feature = interval_path_dic[idx]
            leaf = leafs[idx]
            sample_ids = leaf.get('sample_ids', [])

            disc_feats = samples_disc[sample_ids].T
            cont_feats = samples_cont[sample_ids].T

            intervals_array = np.array(intervals_each_feature, dtype=object)
            intervals_disc = intervals_array[disc_feat_ids]
            intervals_cont = intervals_array[cont_feat_ids]

            if disc_feats.shape[0] > 0:
                disc_samplers = []
                for feat_idx in range(disc_feats.shape[0]):
                    sampler = MultinomialSampler()
                    sampler.fit(disc_feats[feat_idx])
                    disc_samplers.append(sampler)
                MultinomialSampler.generate_new_samples_for_all_features_of_this_type(
                    indices=disc_feat_ids,
                    gen_feats_matrix=gen_feat_matrix,
                    conf_thresh=conf,
                    samplers=disc_samplers
                )

            if cont_feats.shape[0] > 0:
                cont_samplers = []
                if COMBINE_FEAT:
                    sampler = MultivariateGaussianSampler()
                    sampler.fit(cont_feats.T)
                    cont_samplers.append(sampler)
                    MultivariateGaussianSampler.generate_new_samples_for_all_features_of_this_type(
                        indices=cont_feat_ids,
                        gen_feats_matrix=gen_feat_matrix,
                        conf_thresh=conf,
                        samplers=cont_samplers
                    )
                else:
                    for feat_idx in range(cont_feats.shape[0]):
                        sampler = SingleGaussian1DSampler()
                        sampler.fit(cont_feats[feat_idx])
                        cont_samplers.append(sampler)
                    SingleGaussian1DSampler.generate_new_samples_for_all_features_of_this_type(
                        indices=cont_feat_ids,
                        gen_feats_matrix=gen_feat_matrix,
                        conf_thresh=conf,
                        samplers=cont_samplers
                    )

            if all_new_samples.size > 0:
                all_new_samples = np.vstack((all_new_samples, gen_feat_matrix.T))
            else:
                all_new_samples = gen_feat_matrix.T

        return np.clip(all_new_samples, 0, 1)
