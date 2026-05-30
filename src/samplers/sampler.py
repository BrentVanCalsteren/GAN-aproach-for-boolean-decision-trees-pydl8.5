import numpy as np

class Sampler:
    def fit(self, points):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def score_feature(self, feature):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def score_avg(self, feature):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def get_error(self,feature):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def sample(self, n_samples=1, random_state=None):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def sorted_samples(self, n: int) -> np.ndarray:
        raise NotImplementedError('if you wanna use this method you should implement it in child class')
