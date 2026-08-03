from evalData.statistic_sim import feature_corr_diff
from src.usePydl.predictors.local_greedy_predictors import build_tree_iteratively
from data.data_obj.sampels import Samples
import numpy as np
from pydl85 import DL85Classifier
from sklearn.model_selection import train_test_split
from src.usePydl.classifier.ensemble_classifier import EnsembleClassifier
import random
from sklearn.metrics import accuracy_score, classification_report
import CONFIG

#THIS IS A FIRST TEST ON USING PYDL IN DATA GENERATION
# why would we like to use dl trees, well dl algoritm looks at trees differently than the convetional approash
# it will tackle the problem as a itemset mining problem.
# what does item setmining do? it will try to find the most frequent together bought items
# it's clustering items that have a relation with eachother together.
# this can be very effective when using this as a tool in synthetic data generation
# first cluster the samples that have a close relationship with eachother with dltree (each leaf is a cluster)
# and use these clusters for better data generation.
# how the trees work
# Each path in a given tree will be seen as an itemset by DL8.
# instead of using regular number splits that normal trees derive from the features of samples it's trained on,
# dl trees will work with boolean splits, so a feature can only have value 0 or 1.
# this means we will have to convert are numeric features to boolean.
# ======================================================================
# this project has currently only implemented that each numeric feature will be mapped seperatly into
# list of boolean features. Each bool feature can be seen as a split based on a numeric value and or interval
# see feature.py how these get calculated
# after generating some possible good splits, these will be used to create a tree.
# since dl is a form of clustering you can give it costum (error/ distance functions)
# tree (predictors.py) uses default error function (in eror_fun.py) for picking good splits and
# leaf vals will return default leaf val in leaf.py
#==============================================================================
# right now if have to sample functions i use: 1) one that just tries to make each feature interval as small as possible
# 2) one that tries to calc an error by creating a sampler object every time -> is way slower and the results are identical / worse
# (perhaps mulit-feature samplers are better at this, but it would still make it extreemly slow, while in theory it give the same result:
# reducing the intervals)
# error vals are used to score the possible cluster. (best split will be a split that result into 2 clusters with the lowest error)
# now i have also implemented a ensemble tree approach the problem with this is that it can not gerantee that the splits are the best.
# it can be that the split is the best for depth 2 but for depth 4 not. Since the error from split depth 2
# will be calculated based on the erros from splits at depth 3 under him ect.
# (different depths will generate different clusters and does result into different errors)

DO_CLASSIFICATION = True

def test_data_generation():
    sample_obj = Samples(dataset='iris',data_type='tabular')
    sample_obj.load_chunk(0)
    samples = sample_obj.samples
    sample_obj.save_output(samples=samples,llables=None,output_name='original_encoded')
    #==========================
    #create splits
    feature_data = sample_obj.current_feat_hist
    feature_data.creat_splits(total_num_splits=CONFIG.MAX_BOOL_SPLITS)
    # test image convertion
    splits = feature_data.get_splits()
    same_splits = feature_data.splits_obj.map_samples_to_splits(samples=samples)
    print(f'bool convertion works correctly: {np.equal(splits, same_splits).all()}')
    #=========================================
    #now let's test the quality of the splits, good splits will result into good classification with dlclassifier
    labels = sample_obj.labels.flatten()
    train_x,test_x,train_y,test_y = train_test(splits=splits, samples=labels,test_size=0.2)
    classify(train_x,test_x,train_y,test_y)
    #=================================================
    #now let's generate new data

    ensemble_pred = build_tree_iteratively(feature_data)
    samples_gen = ensemble_pred.gen_new_data_based_tree_structure(n=200, conf=0.0)
    #pred = Predictor(samples=samples,splits=splits,max_depth=3,min_sup=1,time=100,n_samples=samples.shape[0])
    #samples_gen = pred.gen_new_data(split_obj=splits_obj, n=150, conf=0.8)
    #=================================================
    #now let's see if classification is better with extra generated data
    splits_gen_x = feature_data.splits_obj.map_samples_to_splits(samples_gen)
    labels_gen = sample_obj.get_best_matching_label(samples=samples_gen, chunk_id=0)
    #test on generated data alone
    classify(splits_gen_x, test_x, labels_gen, test_y)
    splits_combined = np.vstack((train_x, splits_gen_x))
    y_combined = np.hstack((train_y, labels_gen))
    # test on combined data
    classify(splits_combined, test_x, y_combined, test_y)
    sample_obj.save_output(samples=samples_gen, llables=labels_gen,output_name='generated')


def train_test(splits, samples, test_size=0.2):
    splits_train, splits_test, samples_train, samples_test = train_test_split(splits, samples, test_size=test_size,
                                                        random_state=random.randint(1, 100))
    return splits_train, splits_test, samples_train, samples_test

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