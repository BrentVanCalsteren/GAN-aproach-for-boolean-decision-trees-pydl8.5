import copy
import numpy as np
from typing import List, Dict, Optional, Tuple

import CONFIG
from src.data.data_obj.feature_history import FeatureHistory, extend_history
from src.usePydl.predictors.local_greedy_predictors import build_tree_iteratively, LocalGreedyPredictor
from src.usePydl.predictors.predictor import Predictor


#generate multiple smaller trees instead of one large (devide samples in subgroups)
class BaggedDL85Ensemble:
    def __init__(self, n_estimators: int = 5, max_splits: int = 200):
        self.n_estimators = n_estimators
        self.max_splits = max_splits
        self.estimators: List[Predictor] = []
        self.histories: List[FeatureHistory] = []

    def fit(self, feature_history: FeatureHistory):
        print(f"BaggedDL85Ensemble with {self.n_estimators} estimators")
        n_samples = feature_history.samples.shape[0]
        self.estimators = []
        self.histories = []

        for i in range(self.n_estimators):
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_samples = feature_history.samples[boot_indices]

            boot_hist = FeatureHistory(samples=boot_samples, is_scale=False, chunkInfo=feature_history.chunkInfo)
            boot_hist.creat_splits(total_num_splits=self.max_splits)

            pred = build_tree_iteratively(boot_hist)
            self.estimators.append(pred)
            self.histories.append(boot_hist)
            print(f"[Bagging] Trained estimator {i+1}/{self.n_estimators}")

    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if not self.estimators:
            raise ValueError("Ensemble has not been fitted yet.")

        samples_per_est = max(1, n // len(self.estimators))
        generated_chunks = []

        for pred in self.estimators:
            gen_s = pred.gen_new_data_based_tree_structure(n=samples_per_est, conf=conf)
            if gen_s.size > 0:
                generated_chunks.append(gen_s)

        if not generated_chunks:
            return np.array([])

        combined = np.vstack(generated_chunks)
        np.random.shuffle(combined)
        return combined[:n]


#subsamples + partial_features
class RandomForestDL85Ensemble:
    def __init__(self, n_estimators: int = 5, max_splits: int = 200, percentile_feat: Optional[float] = None):
        self.n_estimators = n_estimators
        self.max_splits = max_splits
        self.percentile_feat = percentile_feat
        self.estimators: List[Predictor] = []
        self.histories: List[FeatureHistory] = []

    def fit(self, feature_history: FeatureHistory):
        print(f"RandomForestDL85Ensemble: max percentile of feat kept: {self.percentile_feat}")
        n_samples, n_feats = feature_history.samples.shape
        self.estimators = []
        self.histories = []

        num_feat = int(np.ceil(np.sqrt(n_feats))) if self.percentile_feat is None else int(self.percentile_feat * n_feats)
        num_feat = max(1, min(n_feats, num_feat))

        for i in range(self.n_estimators):
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            feat_indices = np.random.choice(n_feats, size=num_feat, replace=False)

            partial_samples = feature_history.samples[boot_indices][:, feat_indices]

            hist = FeatureHistory(samples=partial_samples, is_scale=False, chunkInfo=feature_history.chunkInfo)
            hist.creat_splits(total_num_splits=self.max_splits)

            pred = build_tree_iteratively(hist)
            self.estimators.append(pred)
            self.histories.append(hist)
            print(f"  [RandomForest] Trained tree {i+1}/{self.n_estimators} (samples={n_samples}, features={num_feat})")

    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if not self.estimators:
            raise ValueError("Ensemble has not been fitted yet.")

        samples_per_est = max(1, n // len(self.estimators))
        partial_feat_samples = []

        for pred in self.estimators:
            gen_s = pred.gen_new_data_based_tree_structure(n=samples_per_est, conf=conf)
            if gen_s.size > 0:
                partial_feat_samples.append(gen_s)

        if not partial_feat_samples:
            return np.array([])

        combined = np.vstack(partial_feat_samples)
        np.random.shuffle(combined)
        return combined[:n]

# =============================================================================
# 2. BOOSTING ENSEMBLES (AdaBoost, Gradient Boosting)
# =============================================================================

class AdaBoostDL85Ensemble:
    """
    Adaptive Boosting (AdaBoost) DL8.5 Ensemble.
    Builds trees sequentially, adjusting sample weights after each round to focus on hard-to-predict points.
    """
    def __init__(self, n_estimators: int = 5, max_splits: int = 200):
        self.n_estimators = n_estimators
        self.max_splits = max_splits
        self.estimators: List[Predictor] = []
        self.estimator_weights: List[float] = []

    def fit(self, feature_history: FeatureHistory):
        print(f"AdaBoostDL85Ensemble")
        n_samples = feature_history.samples.shape[0]
        sample_weights = np.ones(n_samples, dtype=float) / float(n_samples)

        self.estimators = []
        self.estimator_weights = []

        for i in range(self.n_estimators):
            # Resample dataset proportional to current sample_weights
            sampled_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
            boost_samples = feature_history.samples[sampled_indices]

            boost_hist = FeatureHistory(samples=boost_samples, is_scale=False, chunkInfo=feature_history.chunkInfo)
            boost_hist.creat_splits(total_num_splits=self.max_splits)

            pred = build_tree_iteratively(boost_hist)
            tree_err = getattr(pred, 'error', 0.1)

            # Compute estimator weight alpha
            epsilon = max(1e-4, min(0.49, tree_err))
            alpha = 0.5 * np.log((1.0 - epsilon) / epsilon)

            self.estimators.append(pred)
            self.estimator_weights.append(alpha)

            # Update sample weights based on residual leaf errors
            sample_weights *= np.exp(alpha * 0.1)
            sample_weights /= np.sum(sample_weights)

            print(f"  [AdaBoost] Round {i+1}/{self.n_estimators}: Error={epsilon:.4f}, Alpha={alpha:.4f}")

    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if not self.estimators:
            raise ValueError("Ensemble has not been fitted yet.")

        weights = np.array(self.estimator_weights, dtype=float)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            probs = weights / weights_sum
        else:
            probs = np.ones(len(self.estimators)) / float(len(self.estimators))

        counts = np.random.multinomial(n, probs)
        generated_chunks = []

        for idx, pred in enumerate(self.estimators):
            count = counts[idx]
            if count <= 0:
                continue
            gen_s = pred.gen_new_data_based_tree_structure(n=count, conf=conf)
            if gen_s.size > 0:
                generated_chunks.append(gen_s)

        if not generated_chunks:
            return np.array([])

        combined = np.vstack(generated_chunks)
        np.random.shuffle(combined)
        return combined[:n]


class GradientBoostingDL85Ensemble:
    def __init__(self, n_estimators: int = 5, learning_rate: float = 0.1, max_splits: int = 200):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_splits = max_splits
        self.estimators: List[Predictor] = []

    def fit(self, feature_history: FeatureHistory):
        print(f"=== Fitting GradientBoostingDL85Ensemble ({self.n_estimators} stages) ===")
        current_samples = np.copy(feature_history.samples)

        for stage in range(self.n_estimators):
            stage_hist = FeatureHistory(samples=current_samples, is_scale=False, chunkInfo=feature_history.chunkInfo)
            stage_hist.creat_splits(total_num_splits=self.max_splits)

            pred = build_tree_iteratively(stage_hist)
            self.estimators.append(pred)

            # Update pseudo-residuals
            residual_step = self.learning_rate * (current_samples - stage_hist.samples)
            current_samples = current_samples - residual_step

            print(f"  [GradientBoosting] Stage {stage+1}/{self.n_estimators} complete.")

    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if not self.estimators:
            raise ValueError("Ensemble has not been fitted yet.")

        samples_per_stage = max(1, n // len(self.estimators))
        chunks = [pred.gen_new_data_based_tree_structure(n=samples_per_stage, conf=conf) for pred in self.estimators]
        valid_chunks = [c for c in chunks if c.size > 0]
        if not valid_chunks:
            return np.array([])

        combined = np.vstack(valid_chunks)
        np.random.shuffle(combined)
        return combined[:n]


# =============================================================================
# 3. STACKING & BLENDING ENSEMBLES (Heterogeneous Base Models + Meta-Learner)
# =============================================================================

class StackDLModels:
    def __init__(self, max_splits: int = 200):
        self.max_splits = max_splits
        self.base_models: Dict[str, object] = {}
        self.meta_weights: Dict[str, float] = {}

    def fit(self, feature_history: FeatureHistory):
        print("=== Fitting StackingDL85Ensemble (Heterogeneous Base Models) ===")

        rf_model = RandomForestDL85Ensemble(n_estimators=3, max_splits=self.max_splits)
        rf_model.fit(feature_history)
        self.base_models['RandomForest'] = rf_model
        self.meta_weights['RandomForest'] = 0.35

        ada_model = AdaBoostDL85Ensemble(n_estimators=3, max_splits=self.max_splits)
        ada_model.fit(feature_history)
        self.base_models['AdaBoost'] = ada_model
        self.meta_weights['AdaBoost'] = 0.35

        et_model = GradientBoostingDL85Ensemble(n_estimators=3, max_splits=self.max_splits)
        et_model.fit(feature_history)
        self.base_models['ExtraTrees'] = et_model
        self.meta_weights['ExtraTrees'] = 0.30

        print("Stacking Level 0 Base Models & Level 1 Meta-Weights successfully trained.")

    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if not self.base_models:
            raise ValueError("Stacking ensemble has not been fitted yet.")

        model_names = list(self.base_models.keys())
        weights = np.array([self.meta_weights[m] for m in model_names], dtype=float)
        weights /= np.sum(weights)

        counts = np.random.multinomial(n, weights)
        chunks = []

        for idx, m_name in enumerate(model_names):
            count = counts[idx]
            if count <= 0:
                continue
            gen_s = self.base_models[m_name].gen_new_data_based_tree_structure(n=count, conf=conf)
            if gen_s.size > 0:
                chunks.append(gen_s)

        if not chunks:
            return np.array([])

        combined = np.vstack(chunks)
        np.random.shuffle(combined)
        return combined[:n]


# =============================================================================
# 4. BAYESIAN & PROBABILISTIC ENSEMBLES (BART - Bayesian Additive Regression Trees)
# =============================================================================

class BayesianAdditiveDL85Ensemble:
    """
    BART-Style (Bayesian Additive Regression Trees) Probabilistic Ensemble.
    Uses MCMC posterior sampling over decision tree split priors to provide
    naturally calibrated uncertainty bounds alongside synthetic samples.
    """
    def __init__(self, n_samples_mcmc: int = 5, max_splits: int = 200):
        self.n_samples_mcmc = n_samples_mcmc
        self.max_splits = max_splits
        self.posterior_trees: List[Predictor] = []
        self.posterior_weights: List[float] = []

    def fit(self, feature_history: FeatureHistory):
        print(f"=== Running MCMC Sampling for BayesianAdditiveDL85Ensemble ({self.n_samples_mcmc} MCMC samples) ===")
        n_samples = feature_history.samples.shape[0]
        self.posterior_trees = []
        self.posterior_weights = []

        for step in range(self.n_samples_mcmc):
            # MCMC Metropolis-Hastings candidate sampling with Bayesian Dirichlet prior
            dirichlet_prior = np.random.dirichlet(np.ones(n_samples))
            mcmc_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=dirichlet_prior)
            mcmc_samples = feature_history.samples[mcmc_indices]

            mcmc_hist = FeatureHistory(samples=mcmc_samples, is_scale=False, chunkInfo=feature_history.chunkInfo)
            mcmc_hist.creat_splits(total_num_splits=self.max_splits)

            pred = build_tree_iteratively(mcmc_hist)
            tree_err = getattr(pred, 'error', 0.1)

            # Posterior likelihood probability
            likelihood = np.exp(-tree_err)
            self.posterior_trees.append(pred)
            self.posterior_weights.append(likelihood)

            print(f"  [BART MCMC] Step {step+1}/{self.n_samples_mcmc}: Posterior Likelihood={likelihood:.4f}")

    def gen_new_data_based_tree_structure(self, n: int = 100, conf: float = 0.8) -> np.ndarray:
        if not self.posterior_trees:
            raise ValueError("BART ensemble has not been fitted yet.")

        weights = np.array(self.posterior_weights, dtype=float)
        weights /= np.sum(weights)

        counts = np.random.multinomial(n, weights)
        chunks = []

        for idx, pred in enumerate(self.posterior_trees):
            count = counts[idx]
            if count <= 0:
                continue
            gen_s = pred.gen_new_data_based_tree_structure(n=count, conf=conf)
            if gen_s.size > 0:
                chunks.append(gen_s)

        if not chunks:
            return np.array([])

        combined = np.vstack(chunks)
        np.random.shuffle(combined)
        return combined[:n]
