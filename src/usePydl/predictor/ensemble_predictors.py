import math
from typing import List
from src.usePydl.predictor.predictor_obj import Predictor
from src.usePydl.error_fun import predictor_error
from src.usePydl.leaf import default_leaf_val, get_leaf_vals
from src.distributions.single_gaussian import SingleGaussian1D_distr
from src.distributions.bernoulli import Bernoulli_Distr
from src.distributions.multinomial import Multinomial_distr
from src.distributions.uniform import Uniform_distr
import numpy as np

MIN_NUM_SAMPLES = 40

class EnsemblePredictor(Predictor):
    def __init__(self,samples, samples_bin):
        self.child_predictors = {}
        self.samples = samples
        super().__init__(
            samples_bin=samples_bin,
            error_fun=predictor_error(self,samples),
            leaf_val=default_leaf_val(self,samples),
            max_depth=2,
            min_sup=1,
            time=100
        )
        self.generate_tree()
        self.generate_child_preds()


    def generate_child_preds(self):
        leafs = get_leaf_vals(self.predictor.tree_)
        samplesIDs_per_leaf = [leaf["value"]["sample_ids"] for leaf in leafs]
        for i, samplesIDs in enumerate(samplesIDs_per_leaf):
            if len(samplesIDs) >= MIN_NUM_SAMPLES and len(samplesIDs) != self.samples.shape[0]:
                print("Have enough samples to generate extra predictors")
                sub_samples = np.array(self.samples[samplesIDs])
                sub_samples_bin = np.array(self.samples_bin[samplesIDs])
                self.child_predictors.update({i:EnsemblePredictor(sub_samples, sub_samples_bin)})

    def get_distributions(self, features:np.ndarray):
        distr_funs = []
        for feature in features:
            distr_funs.append(self.get_distr(feature))
        return distr_funs

    def generate_new_data(self, n_new_samples=100, conf_tresh=0.8, mode: str = "keep_counts") -> np.ndarray:
        new_samples = []
        leafs = get_leaf_vals(self.predictor.tree_)
        distributions_x_leafs = [leaf["value"]["distr"] for leaf in leafs]
        samples_in_leaf = np.array([leaf["value"]["count"] for leaf in leafs])
        total_count = samples_in_leaf.sum()

        if mode == "keep_counts":
            ns = ((samples_in_leaf / total_count) * n_new_samples).astype(int)
        elif mode == "even":
            ns = np.full(len(leafs), n_new_samples // len(leafs))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        for i in range(len(leafs)):
            if i in self.child_predictors:
                print(f"generating samples from child{ns[i]}")
                new_samples.extend(self.child_predictors[i].generate_new_data(
                    n_new_samples=ns[i],
                    conf_tresh=conf_tresh,
                    mode=mode))
            else: new_samples.extend(self._generate_new_leaf_samples(ns[i], distributions_x_leafs[i], conf_tresh))

        return np.array(new_samples)


    def get_distr(self,feature_array:np.ndarray):
        #dist = Uniform_distr()
        dist = SingleGaussian1D_distr()
        dist.fit(feature_array.reshape(-1, 1))
        return dist

    def calc_norm_conf_each_sample(self, distributions, samples):
        features = samples.T  # (n_features, n_samples)
        log_likelihoods = sum(
            gm.score_feature(features[i])
            for i, gm in enumerate(distributions)
        )
        log_likelihoods_max = sum(
            float(gm.score_feature(np.array([gm.mean])))
            for gm in distributions
        )
        return np.exp(log_likelihoods - log_likelihoods_max)

    # used in default error function
"""
    def calc_norm_conf_each_sample(self, distributions: List[Uniform_distr], samples: np.ndarray):
        features = samples.T
        scores = np.array([dist.score_feature(features[i]) for i, dist in enumerate(distributions)]).T #shape scores (n_samples,n_features)
        return np.mean(scores, axis=1)

    """

#"""