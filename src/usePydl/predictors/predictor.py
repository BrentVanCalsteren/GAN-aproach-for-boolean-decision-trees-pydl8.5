from typing import List

import numpy as np
from pydl85 import DL85Predictor
from src.usePydl.error_fun import IntervalSizesError, MSEError, DiameterError
from src.usePydl.leaf import ReturnIDSandPROB
from src.samplers.multinomial import MultinomialSampler
from src.samplers.uniform import UniformSampler
from src.samplers.single_gaussian import SingleGaussian1DSampler
from src.samplers.multivariate_gaussian import MultivariateGaussianSampler
from typing import List

from usePydl.predictors.tree import Tree

COMBINE_FEAT = False


class Predictor:
    n_samples = None

    def __init__(self, splits, samples, max_depth, min_sup, time, n_samples=None):
        print('starting dl predictor...')
        if n_samples is not None:
            self.n_samples = n_samples
        else:
            self.n_samples = samples.shape[0]

        error = IntervalSizesError(samples)
        leaf_val = ReturnIDSandPROB(self.n_samples)
        self.error = error.good_error
        self.dl_predictor = DL85Predictor(error_function=error,
                                          leaf_value_function=leaf_val,
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)

        self.dl_predictor.fit(splits)
        self.tree = Tree(tree=self.dl_predictor.tree_)


    def predict(self, samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def get_tree_dict(self):
        return self.tree.tree

    def gen_new_data(self, split_obj=None, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if split_obj is None:
            raise ValueError('split_obj is None')
        self.tree.feature_index_array = split_obj.feature_index_array

        leafs = self.tree.get_leafs()
        if len(leafs) == 0:
            raise ValueError("Tree has no leaves.")

        interval_path_dic = self.tree.get_intervals_each_path()
        samples = split_obj.sample_obj.samples
        feat_info = split_obj.sample_obj.get_feature_info()

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
        sample_count = 0

        for idx, count in zip(indx, counts):
            gen_feat_matrix = np.zeros((len(feat_info), count))
            intervals_each_feature = interval_path_dic[idx]
            leaf = leafs[idx]
            sample_ids = leaf['value'].get('sample_ids', [])

            if len(sample_ids) == 0:
                print(f"Warning: Selected leaf {idx} has 0 samples. Skipping.")
                continue

            sample_count += count
            disc_feats = samples_disc[sample_ids].T
            cont_feats = samples_cont[sample_ids].T

            intervals_disc = []
            intervals_cont = []
            for i in range(len(feat_info)):
                if disc_feat_ids[i]:
                    intervals_disc.append(intervals_each_feature[i])
                else:
                    intervals_cont.append(intervals_each_feature[i])

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
                    samplers=disc_samplers,
                    intervals_list=intervals_disc
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
                        samplers=cont_samplers,
                        intervals_list=intervals_cont
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
                        samplers=cont_samplers,
                        intervals_list=intervals_cont
                    )

            if all_new_samples.size > 0:
                all_new_samples = np.vstack((all_new_samples, gen_feat_matrix.T))
            else:
                all_new_samples = gen_feat_matrix.T

        print(f"Generated {sample_count} samples.")
        if all_new_samples.size == 0:
            return all_new_samples
        return np.clip(all_new_samples, 0, 1)

