from typing import List
from pydl85 import DL85Predictor
from sklearn.mixture import GaussianMixture
from src.usePydl.error_fun import mse_error
import numpy as np

from src.usePydl.leaf import empty_leave_val


class DefaultPredictor:
    """
    use default pydl, cluster algoritm
    """
    def __init__(self,samples, samples_bin, max_depth=3,min_sup=1,time=100):
        self.samples = samples
        self.samples_bin = samples_bin
        self.predictor =  DL85Predictor(
            max_depth=max_depth,
            min_sup=min_sup,
            error_function=mse_error(samples),
            leaf_value_function=empty_leave_val(),
        )
        self.generate_tree()


    def generate_tree(self):
        self.predictor.fit(self.samples_bin)

    def get_distr(self, feature_array):
        return

    def calc_norm_conf_each_sample(self, distributions: List[GaussianMixture], samples):
        return
