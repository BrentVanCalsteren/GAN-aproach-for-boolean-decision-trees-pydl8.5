from src.usePydl.predictor.predictor import Predictor
from src.usePydl.leaf import get_leaf_vals
import numpy as np

MIN_NUM_SAMPLES = 10


class EnsemblePredictor(Predictor):
    def __init__(self,splits, samples, sampler_types):
        print(f"Starting Ensemble Predictor on {len(samples)} samples...")
        self.child_predictors = {}
        super().__init__(
            splits=splits,
            samples=samples,
            sampler_types=sampler_types,
            max_depth=2,
            min_sup=1,
            time=100
        )
        self.generate_child_preds(samples, splits, sampler_types)


    def generate_child_preds(self,samples,splits, sampler_types):
        leafs = get_leaf_vals(self.dl_predictor.tree_)
        samplesIDs_per_leaf = [leaf["value"]["sample_ids"] for leaf in leafs]
        for i, samplesIDs in enumerate(samplesIDs_per_leaf):
            if len(samplesIDs) >= MIN_NUM_SAMPLES and len(samplesIDs) != samples.shape[0]:
                sub_samples = np.array(samples[samplesIDs])
                sub_splits = np.array(splits[samplesIDs])
                print(f"Have enough samples {len(samplesIDs)} to generate extra predictors, num samples: {len(samplesIDs)}")
                self.child_predictors.update({i:EnsemblePredictor(sub_splits, sub_samples, sampler_types)})


    def generate_new_data(self, n_new_samples=100, conf_tresh=0.8, mode: str = "keep_counts") -> np.ndarray:
        new_samples = []
        leafs = get_leaf_vals(self.dl_predictor.tree_)
        samplers_x_leafs = [leaf["value"]["samplers_list"] for leaf in leafs]
        samples_in_leaf = np.array([leaf["value"]["count"] for leaf in leafs])
        total_count = samples_in_leaf.sum()

        if mode == "keep_counts":
            ns = ((samples_in_leaf / total_count) * n_new_samples).astype(int) + 1
        elif mode == "even":
            ns = np.full(len(leafs), n_new_samples // len(leafs)) + 1
        else:
            raise ValueError(f"Unknown mode: {mode}")
        for i in range(len(leafs)):
            if i in self.child_predictors:
                print(f"generating samples from child{ns[i]}")
                new_samples.extend(self.child_predictors[i].generate_new_data(
                    n_new_samples=ns[i],
                    conf_tresh=conf_tresh,
                    mode=mode))
            else: new_samples.extend(self._generate_new_leaf_samples(ns[i], samplers_x_leafs[i], conf_tresh))

        return np.array(new_samples)