from src.usePydl.predictors.local_greedy_predictors import build_tree_iteratively
from data.data_obj.sampels import Samples
import numpy as np
from pydl85 import DL85Classifier
from src.usePydl.classifier.ensemble_classifier import EnsembleClassifier
from sklearn.metrics import accuracy_score, classification_report
from usePydl.predictors.random_forest_predictor import RandomForestPredictor

DO_CLASSIFICATION = False
def test_data_generation():
    sample_obj = Samples(dataset='mnist_jpeg',data_type='image',labels_at_front=False)
    sample_obj.load_chunk(chunk_id=0,split_test=0.2)
    samples_train = sample_obj.samples
    samples_test = sample_obj.samples_test
    sample_obj.save_output(samples=samples_train,llables=None,output_name='original_encoded')

    #==========================
    #create splits
    feature_data = sample_obj.current_feat_hist
    feature_data.stays_in_memory = True
    feature_data.creat_splits()
    # test image convertion
    train_splits = feature_data.get_splits()
    same_splits = feature_data.splits_obj.map_samples_to_splits(samples=samples_train)
    print(f'bool convertion works correctly: {np.equal(train_splits, same_splits).all()}')
    test_splits = feature_data.splits_obj.map_samples_to_splits(samples=samples_test)

    #=========================================
    #now let's test the quality of the splits, good splits will result into good classification with dlclassifier
    labels_train = sample_obj.labels.flatten()
    labels_test = sample_obj.labels_test.flatten()
    classify(train_splits,test_splits,labels_train,labels_test)
    #=================================================
    #now let's generate new data

    #pred = build_tree_iteratively(feature_data)
    pred = RandomForestPredictor(sample_obj, max_features_per_tree=2)
    samples_gen = pred.gen_new_data(n=200, conf=0.5)
    #pred = Predictor(samples=samples,splits=splits,max_depth=3,min_sup=1,time=100,n_samples=samples.shape[0])
    #samples_gen = pred.gen_new_data(split_obj=splits_obj, n=150, conf=0.8)
    #=================================================
    #now let's see if classification is better with extra generated data
    splits_gen_x = feature_data.splits_obj.map_samples_to_splits(samples_gen)
    labels_gen = sample_obj.get_best_matching_labels(gen_samples=samples_gen, chunk_id=0)
    #test on generated data alone
    classify(splits_gen_x, test_splits, labels_gen, labels_test)
    splits_combined = np.vstack((train_splits, splits_gen_x))
    y_combined = np.hstack((labels_train, labels_gen))
    # test on combined data
    classify(splits_combined, test_splits, y_combined, labels_test)
    sample_obj.save_output(samples=samples_gen, llables=labels_gen,output_name='generated')


def classify_ensemble(x_train, x_test, y_train, y_test):
    if DO_CLASSIFICATION:
        print("Running pydl classifier")
        clasfi = EnsembleClassifier(max_depth=3, min_sup=1, time_limit=100)
        clasfi.fit(x_train,y_train)
        y_pred_test = clasfi.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

def classify(x_train, x_test, y_train, y_test):
    if DO_CLASSIFICATION:
        depth = 1
        uniques = np.unique(y_train)
        while 2**(depth-1) < len(uniques):
            depth += 1
        print("Running pydl classifier")
        clasfi = create_classifier_default(
            x_bin=x_train, y=y_train, max_depth=depth, min_sup=1, time=100)
        y_pred_test = clasfi.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

def create_classifier_default(x_bin,y,max_depth=3,min_sup=2,time=100):
    clasfi = DL85Classifier(max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(x_bin, y)
    return clasfi

def value_to_index(array):
    unique_values = np.unique(array)
    indeces = np.zeros(array.shape)
    for i, val in enumerate(unique_values):
        indeces[array==val] = i
    return indeces



if __name__ == "__main__":
    test_data_generation()

"""
OUTPUT EXEMPLE WITH DEFAULT ERROR (sampler method
=========================================
last outputs of classifier: (tested on iris dataset) (runtime around 10 minutes)
on data alone:
Accuracy: 0.8000

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      0.78      0.88         9
         1.0       0.62      1.00      0.77        10
         2.0       1.00      0.64      0.78        11

    accuracy                           0.80        30
   macro avg       0.88      0.80      0.81        30
weighted avg       0.88      0.80      0.80        30
-------------------------------------------------------
on gen data alone:
Accuracy: 0.9000

Classification Report:
              precision    recall  f1-score   support

         0.0       0.75      1.00      0.86         9
         1.0       1.00      0.90      0.95        10
         2.0       1.00      0.82      0.90        11

    accuracy                           0.90        30
   macro avg       0.92      0.91      0.90        30
weighted avg       0.93      0.90      0.90        30
-------------------------------------------------------
combined:
Accuracy: 0.9667

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00         9
         1.0       0.91      1.00      0.95        10
         2.0       1.00      0.91      0.95        11

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30

#OUTPUT WITH MIN_INTERVAL ERROR (same settings except error fun)
========================================
last outputs of classifier: (tested on iris dataset) (runtime like 10 seconds = 60 times faster)
on data alone:

Accuracy: 0.9333

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00         7
         1.0       0.88      1.00      0.93        14
         2.0       1.00      0.78      0.88         9

    accuracy                           0.93        30
   macro avg       0.96      0.93      0.94        30
weighted avg       0.94      0.93      0.93        30

-------------------------------------------------------
on gen data alone:
Accuracy: 0.9000

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      0.86      0.92         7
         1.0       0.87      0.93      0.90        14
         2.0       0.89      0.89      0.89         9

    accuracy                           0.90        30
   macro avg       0.92      0.89      0.90        30
weighted avg       0.90      0.90      0.90        30

-------------------------------------------------------
combined:
Accuracy: 0.9667

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00         7
         1.0       0.93      1.00      0.97        14
         2.0       1.00      0.89      0.94         9

    accuracy                           0.97        30
   macro avg       0.98      0.96      0.97        30
weighted avg       0.97      0.97      0.97        30

"""