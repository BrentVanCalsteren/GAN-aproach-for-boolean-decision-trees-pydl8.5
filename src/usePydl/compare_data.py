from src.dataLoader.Dataset import *
from sklearn.model_selection import train_test_split
import random
import classifier_pydl
from sklearn.metrics import accuracy_score, classification_report


def compare_data_with_pydl_classifier():
    data_set = dataset(dataset_name='iris',bin_length=-1)
    data_set.split_data_on_features(n_splits=2)
    datas = data_set.get_data_at_split_depth(1)
    l = len(datas)
    predictor_types = ["gaussian"] * l
    for data in datas:
        print(data.x.shape)
    data_set.load_predictors(datas,predictor_types)
    data_set.gen_new_samples_for_datalist(datas, 100, 0.8)
    data_set.gen_data_for_parents(datas)
    data_level_0 = data_set.root_data
    x_bin = data_level_0.x_bin[:,:-data_level_0.feature_bin_len[-1]]
    y = data_level_0.x[:,-1]
    X_train, X_test, y_train, y_test = split_train_test(x_bin, value_to_index(y), test_size=0.2)
    classify_test_pydl(X_train, X_test, y_train, y_test)
    x_gen_bin = data_level_0.x_gen_bin[:,:-data_level_0.feature_bin_len[-1]]
    y_gen = _map_array_to_closest_val(y,data_level_0.x_gen[:,-1])
    classify_test_pydl(x_gen_bin,x_bin,value_to_index(y_gen),value_to_index(y))

def classify_test_pydl(x_train, x_test, y_train, y_test):
    clasfi = classifier_pydl.classify_with_default_error(
        x_bin=x_train, y=y_train, max_depth=4, min_sup=1, time=300)
    y_pred_test = clasfi.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

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


def split_train_test(X, Y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                        random_state=random.randint(1, 100))
    return X_train, X_test, y_train, y_test

#TODO: can nns be used? (normally for images) Fréchet Inception Distance (FID), Inception Score (IS) and Kernel Inception Distance (KID) be used for evaluating?
#todo: check Statistical Similarity: Univariate Distribution, Bivariate Correlation, Multivariate Distribution, Propensity Mean Squared Error (pMSE)

if __name__ == '__main__':
    compare_data_with_pydl_classifier()