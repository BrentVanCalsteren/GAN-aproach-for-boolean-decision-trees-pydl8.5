from src.usePydl.predictor.predictor_obj import Predictor
from src.usePydl.leaf import leaf_val_gaussian_distributions
from src.usePydl.error_fun import prob_norm_error2
import src.chanceCalc.prob_estimator as probEstimator
import numpy as np

class GaussianPredictor(Predictor):
    def __init__(self,data,max_depth=3,min_sup=2,time=100):
        super().__init__(
            data=data,
            error_fun=prob_norm_error2(data.x),
            leaf_val=leaf_val_gaussian_distributions(data.x),
            max_depth=max_depth,
            min_sup=min_sup,
            time=time
        )

    def generate_new_data(self,conf_trash=0.8,number_of_new_samples=100):
        #todo: improve
        leafs = self.get_leaf_vals()
        distrbutions_leafs = []
        errors = []
        for leaf in leafs:
            distrbutions_leafs.append(leaf['value'])
            errors.append(leaf['error'])
        new_samples = []
        while len(new_samples) < number_of_new_samples:
            z = np.random.random(size=(100, self.data.x.shape[1]))
            if self.data.is_shifted:
                z += 1
            z_scores = np.array([0.0] * number_of_new_samples)
            for i, distributions in enumerate(distrbutions_leafs):
                z_prob_matrix = probEstimator.calc_normalised_confidence_gaussian(distributions, z)
                for j in range(number_of_new_samples):
                    if z_scores[j] < np.min(z_prob_matrix[j]):
                        z_scores[j] = np.min(z_prob_matrix[j])
            indices = np.argsort(z_scores)[-10:]
            candidates = z[indices]
            scores = z_scores[indices]
            for i, candidate in enumerate(candidates):
                if scores[i] > conf_trash:
                    new_samples.append(candidate)
                    print(f'found sample with score{scores[i]}')
        return np.array(new_samples)

