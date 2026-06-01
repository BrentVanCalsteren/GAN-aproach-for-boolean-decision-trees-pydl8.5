from sympy.codegen.ast import none

from src.usePydl.predictor.ensemble_predictors import EnsemblePredictor
from src.data.sampels import Samples
import numpy as np
from pydl85 import DL85Classifier
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, classification_report

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
# tree (predictor.py) uses default error function (in eror_fun.py) for picking good splits and
# leaf vals will return default leaf val in leaf.py
#==============================================================================
# right now samplers will be created in error function based on the numeric values of samples (each sampler is based on a different distribution (all work on a single feat))
# these samplers will be used to score the possible cluster. (best split will be a split that result into 2 clusters with the lowest error)
# now i have implemented a ensemble tree approach the problem with this is that it can not gerantee that the splits are the best.
# it can be that the split is the best for depth 2 but for depth 4 not. Since the error from split depth 2
# will be calculated based on the erros from splits at depth 3 under him ect.
# (different depths will generate different clusters and does result into different errors)

def test_data_generation():
    sample_obj = Samples('iris')
    #==========================
    #first simple test thats dataset load and generating splits work correctly
    sample_obj.creat_splits(total_split_num=90)
    splits = sample_obj.get_splits()
    samples = sample_obj.get_samples()
    same_splits = sample_obj.map_other_samples_to_same_splits(samples) #this is to check if the function works for mapping other samples features to the same boolean splits.
    print(f'bool convertion works correctly: {np.equal(splits, same_splits).all()}')
    #=========================================
    #now let's test the quality of the splits, good splits will result into good classification with dlclassifier
    splits_x = sample_obj.get_splits(slices=slice(0,-1))
    samples_y = sample_obj.get_samples(slices=slice(-1,None,None),convert_to_int=True) #convert back to int, dl classifier needs int labels
    train_x,test_x,train_y,test_y = train_test(splits=splits_x, samples=samples_y,test_size=0.2)
    classify_test_pydl(train_x,test_x,train_y,test_y)
    #=================================================
    #now let's generate new data
    ensemble_pred = EnsemblePredictor(splits, samples,sample_obj.get_feature_types())
    samples_gen = ensemble_pred.generate_new_data(n_new_samples=200, conf_tresh=0.8) #conf_tresh is how high the features for each sample need to score
    #=================================================
    #now let's see if classification is better with extra generated data
    splits_gen = sample_obj.map_other_samples_to_same_splits(samples_gen,slices=slice(0,-1))
    y_gen = value_to_index(samples_gen[:, -1])
    #test on generated data alone
    classify_test_pydl(splits_gen, test_x, y_gen, test_y)
    splits_combined = np.vstack((train_x, splits_gen))
    y_combined = np.hstack((train_y, y_gen))
    # test on combined data
    classify_test_pydl(splits_combined, test_x, y_combined, test_y)


def train_test(splits, samples, test_size=0.2):
    splits_train, splits_test, samples_train, samples_test = train_test_split(splits, samples, test_size=test_size,
                                                        random_state=random.randint(1, 100))
    return splits_train, splits_test, samples_train, samples_test

def classify_test_pydl(x_train, x_test, y_train, y_test):

    depth = 1
    uniques = np.unique(y_train)
    while 2**(depth) < x_train.shape[0]:
        depth += 1
    print("Running pydl classifier")
    clasfi = create_classifier_default(
        x_bin=x_train, y=y_train, max_depth=depth, min_sup=1, time=300)
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
last outputs of classifier: (always tested on iris dataset)
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
"""