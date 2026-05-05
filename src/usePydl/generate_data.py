from src.dataLoader.Dataset import *

def open_dataset(pred_type=None, try_splits = 1,dataset_name="iris",time=100):
    data_set = dataset(dataset_name=dataset_name, bin_length=-1)
    data_set.split_data_on_features(n_splits=try_splits)
    data_set.set_active_depth(try_splits)
    if pred_type:
        data_set.load_predictors(predictor_types=pred_type, max_depth=3, time=time)
    else: data_set.load_predictors(predictor_types="uniform", max_depth=3, time=time)
    return data_set

def _map_array_to_closest_val(vals, array):
    unique_values = np.unique(vals)
    y_mapped = np.zeros(array.shape)
    for i, y in enumerate(array):
        y_mapped[i] = np.argsort(np.abs(unique_values - y))[0]
    return y_mapped


#TODO: can nns be used? (normally for images) Fréchet Inception Distance (FID), Inception Score (IS) and Kernel Inception Distance (KID) be used for evaluating?
#todo: check Statistical Similarity: Univariate Distribution, Bivariate Correlation, Multivariate Distribution, Propensity Mean Squared Error (pMSE)
