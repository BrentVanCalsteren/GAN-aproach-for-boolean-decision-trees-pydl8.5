import numpy as np
from pydl85 import DL85Predictor

from data.data_obj import feature_history
from src.usePydl.error_fun import IntervalSizesError, MSEError, DiameterError
from src.usePydl.leaf import ReturnIDSandPROB
from src.samplers.multinomial import MultinomialSampler
from src.samplers.uniform import UniformSampler
from src.samplers.single_gaussian import SingleGaussian1DSampler
from src.samplers.multivariate_gaussian import MultivariateGaussianSampler
import CONFIG
from usePydl.error_fun import CombinedMSEIntervalError

from usePydl.predictors.tree import Tree, remap_tree

COMBINE_FEAT = False


class Predictor:
    n_samples = None

    def __init__(self, feat_hist, weights, max_depth, min_sup, time):
        if weights is None:
            weights = feat_hist.chunkInfo.feature_importance
        print('starting dl predictor...')
        error_fun = IntervalSizesError(feat_hist.samples, weights)
        leaf_val = ReturnIDSandPROB(feat_hist.get_first_hist_depth_0().samples.shape[0])
        self.dl_predictor = DL85Predictor(error_function=error_fun,
                                          leaf_value_function=leaf_val,
                                          max_depth=max_depth,
                                          min_sup=min_sup,
                                          time_limit=time,
                                          max_error=np.inf)

        self.dl_predictor.fit(feat_hist.get_splits())
        self.error = self.dl_predictor.error_
        tree = Tree(tree=self.dl_predictor.tree_)
        tree.tree = remap_tree(tree.tree, feature_history=feat_hist)
        feat_hist.tree = tree
        self.tree = tree
        self.feature_history = feat_hist


    def predict(self, samples_bin):
        return self.dl_predictor.predict(samples_bin)

    def get_tree_dict(self):
        complete = self.feature_history.get_complete_tree()
        return complete.tree if complete is not None else self.tree.tree

    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        complete_tree = self.feature_history.get_complete_tree()
        leafs = complete_tree.get_leafs()
        if len(leafs) == 0:
            raise ValueError("Tree has no leaves")

        feat_info_list = self.feature_history.feature_info_list
        print(f'a path: {complete_tree.get_all_paths()[0]}')
        print(f'avg feat distribution of path: {complete_tree.avg_features_used_each_path(len(feat_info_list))}')
        interval_path_dic = complete_tree.get_intervals_each_path(feat_history=self.feature_history)
        cont_disc_feats = ['continuous'] * len(feat_info_list)
        cont_feat_ids = [t == 'continuous' for t in cont_disc_feats]

        probs_each_path = np.array([leaf['value']['rel_prob'] for leaf in leafs])
        prob_sum = np.sum(probs_each_path)
        if prob_sum > 0:
            probs_each_path /= prob_sum
        else:
            probs_each_path = np.ones(len(probs_each_path)) / len(probs_each_path)

        all_new_samples = np.array([])
        indx_list_n = np.random.choice(len(leafs), n, p=probs_each_path)
        indx, counts = np.unique(indx_list_n, return_counts=True)
        sample_count = 0

        for idx, count in zip(indx, counts):
            gen_feat_matrix = np.zeros((len(feat_info_list), count))
            intervals_each_feature = interval_path_dic[idx]
            sample_count += count

            intervals_disc = []
            intervals_cont = []
            for i in range(len(feat_info_list)):
                intervals_cont.append(intervals_each_feature[i])


            if len(intervals_disc) > 0:
                disc_samplers = []
                for intervals_feat in intervals_disc:
                    sampler = MultinomialSampler()
                    interval = intervals_feat.get_complete_domain()
                    sampler.fit(np.array(interval))
                    disc_samplers.append(sampler)
                MultinomialSampler.generate_new_samples_for_all_features_of_this_type(
                    indices=[],
                    gen_feats_matrix=gen_feat_matrix,
                    conf_thresh=conf,
                    samplers=disc_samplers,
                    intervals_list=intervals_disc
                )

            if len(intervals_cont) > 0:
                samplers_for_feat = []
                for intervals_feat in intervals_cont:
                    interval = intervals_feat.get_complete_domain()
                    sampler = SingleGaussian1DSampler()
                    sampler.fit(np.array(interval))
                    samplers_for_feat.append(sampler)
                SingleGaussian1DSampler.generate_new_samples_for_all_features_of_this_type(
                    indices=cont_feat_ids,
                    gen_feats_matrix=gen_feat_matrix,
                    conf_thresh=conf,
                    samplers=samplers_for_feat,
                    intervals_list=intervals_cont
                )
            if all_new_samples.size > 0:
                all_new_samples = np.vstack((all_new_samples, gen_feat_matrix.T))
            else:
                all_new_samples = gen_feat_matrix.T

        print(f"Generated {sample_count} samples.")
        if all_new_samples.size == 0:
            return all_new_samples
        return all_new_samples

