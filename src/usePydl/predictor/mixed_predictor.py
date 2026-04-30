import math
from typing import List
from src.usePydl.predictor.predictor_obj import Predictor
from src.usePydl.error_fun import predictor_error
from src.usePydl.leaf import default_leaf_val
from src.distributions.single_gaussian import SingleGaussian1D_distr
from src.distributions.bernoulli import Bernoulli_Distr
from src.distributions.multinomial import Multinomial_distr
import numpy as np

class MixedPredictor(Predictor):
    def __init__(self,samples, samples_bin,discrete_feature_ids, max_depth=3,min_sup=1,time=100):
        self.samples = samples
        self.discrete_feature_ids = discrete_feature_ids
        super().__init__(
            samples_bin=samples_bin,
            error_fun=predictor_error(self,samples),
            leaf_val=default_leaf_val(self,samples),
            max_depth=max_depth,
            min_sup=min_sup,
            time=time
        )
        self.generate_tree()


    def get_distributions(self, features:np.ndarray):
        distr_funs = []
        for i in range(features.shape[0]):
            if i in self.discrete_feature_ids:
                if len(np.unique(features[i])) == 2:
                    distr_funs.append(self.get_distr("bernoulli",features[i]))
                else:
                    distr_funs.append(self.get_distr("multinomial",features[i]))
            else:
                distr_funs.append(self.get_distr("gaussian",features[i]))
        return distr_funs

    def get_distr(self, type:str, feature_array:np.ndarray):
        if type == "bernoulli":
            dist = Bernoulli_Distr()
            dist.fit(feature_array.reshape(-1,1))
            return dist
        elif type == "multinomial":
            dist = Multinomial_distr()
            dist.fit(feature_array.reshape(-1,1))
            return dist
        elif type == "gaussian":
            dist = SingleGaussian1D_distr()
            dist.fit(feature_array.reshape(-1,1))
            return dist
        else:
            raise ValueError("type does not match disctribution type")

    # used in default error function
    def calc_norm_conf_each_sample(self, distributions, samples:np.ndarray): #returns conf of each sample
        features = samples.T  # (n_features, n_samples)
        conf = np.zeros(samples.shape[0]) # n_samples
        for i, distribution in enumerate(distributions):
            score = distribution.score_feature(features[i])
            if isinstance(distribution, SingleGaussian1D_distr):
                max_ll = float(distribution.score_feature(np.array([distribution.mean])))
                conf += np.exp(score - max_ll)
            elif isinstance(distribution, Bernoulli_Distr):
                conf += score
            elif isinstance(distribution, Multinomial_distr):
                conf += score
        return conf
