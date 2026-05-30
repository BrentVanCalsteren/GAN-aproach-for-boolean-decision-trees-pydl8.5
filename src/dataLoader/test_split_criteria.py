from src.usePydl.predictor.ensemble_predictors import EnsemblePredictor
from src.evalData.eval_data import *
from dataLoader.sampels import Samples


def test_split():
    sample_obj = Samples('iris')
    samples = sample_obj.get_samples()
    bool_splits, bool_len_each_feat = sample_obj.find_best_splits()
    x = samples[:,:-1]
    x_bin = bool_splits[:,:-bool_len_each_feat[-1]]
    print(x_bin,x)
    y = samples[:,-1]
    y = value_to_index(y)
    y_bin = bool_splits[:,-bool_len_each_feat[-1]]
    x_train, x_test, y_train, y_test = split_train_test(x_bin,y,0.2)
    classify_test_pydl(x_train, x_test, y_train, y_test)
    ensemble_pred = EnsemblePredictor(samples, bool_splits)
    samples_gen = ensemble_pred.generate_new_data(n_new_samples=100, conf_tresh=0.8)
    samples_gen = make_discrete(samples,samples_gen)
    samples_gen_bin = sample_obj.map_other_samples_to_same_splits(samples_gen)
    y_gen = samples_gen[:, -1]
    y_gen = value_to_index(y_gen)
    x_gen_bin = samples_gen_bin[:, :-bool_len_each_feat[-1]]
    classify_test_pydl(x_gen_bin, x_bin, y_gen, y)









def value_to_index(array):
    unique_values = np.unique(array)
    indeces = np.zeros(array.shape)
    for i, val in enumerate(unique_values):
        indeces[array==val] = i
    return indeces


def make_discrete(x, x_gen):
    features_gen = x_gen.T
    features_real = x.T
    converted_features = []
    for i,feature in enumerate(features_gen):
        uniques = np.unique(features_real[i])
        if len(uniques) < len(features_real[i])*5//100:
            print(f"discritisising gen feature: {i}")
            conv_feature = []
            for f in feature:
                dists = np.abs(uniques - f)
                conv_feature.append(uniques[np.argmin(dists)])
            converted_features.append(conv_feature)
        else: converted_features.append(feature)
    return np.array(converted_features).T





if __name__ == "__main__":
    test_split()