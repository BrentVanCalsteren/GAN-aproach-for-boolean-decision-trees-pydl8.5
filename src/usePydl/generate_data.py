from src.dataLoader.Dataset import *

def generate_new_data(pred_type=None, try_splits = 1,n_samples= 1000,dataset_name="iris",y_index = None):
    data_set = dataset(dataset_name=dataset_name, bin_length=-1)
    data_set.split_data_on_features(n_splits=try_splits)
    datas = data_set.get_data_at_split_depth(try_splits)
    if not pred_type:
        pred_types = ["uniform"] * len(datas)
    else:
        pred_types = [pred_type] * len(datas)
    data_set.load_predictors(datas, pred_types, max_depth=4, time=100)
    data_set.gen_new_samples_for_datalist(datas, n_samples, 0.9)
    data_level_0 = data_set.root_data
    if y_index:
        x, x_bin, x_gen, x_gen_bin, y, y_gen = split_x_y(data_level_0,y_index)
        return x, x_bin, x_gen, x_gen_bin, y, y_gen
    else:
        x_bin = data_level_0.x_bin
        x = data_level_0.x
        x_gen = data_level_0.x_gen
        x_gen_bin = data_level_0.x_gen_bin
        return x, x_bin, x_gen, x_gen_bin, None, None

def split_x_y(data_level_0,y_index):
    x = data_level_0.x[:,:y_index]
    x_bin = data_level_0.x_bin[:,:-data_level_0.feature_bin_len[y_index]]
    x_gen = data_level_0.x_gen[:,:y_index]
    x_gen_bin = data_level_0.x_gen_bin[:,:-data_level_0.feature_bin_len[y_index]]
    y_real = data_level_0.x[:, y_index]
    y_gen = _map_array_to_closest_val(y_real,data_level_0.x_gen[:,y_index])
    y = value_to_index(y_real)
    y_gen = value_to_index(y_gen)
    return x, x_bin, x_gen, x_gen_bin, y, y_gen


def value_to_index(array):
    unique_values = np.unique(array)
    indeces = np.zeros(array.shape)
    for i, val in enumerate(unique_values):
        indeces[array==val] = i
    return indeces


def _map_array_to_closest_val(vals, array):
    unique_values = np.unique(vals)
    y_mapped = np.zeros(array.shape)
    for i, y in enumerate(array):
        y_mapped[i] = np.argsort(np.abs(unique_values - y))[0]
    return y_mapped


#TODO: can nns be used? (normally for images) Fréchet Inception Distance (FID), Inception Score (IS) and Kernel Inception Distance (KID) be used for evaluating?
#todo: check Statistical Similarity: Univariate Distribution, Bivariate Correlation, Multivariate Distribution, Propensity Mean Squared Error (pMSE)
