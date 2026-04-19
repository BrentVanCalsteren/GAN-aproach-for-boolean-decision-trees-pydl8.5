from src.dataLoader.Dataset import *
import src.usePydl.leaf as l
import predictor.gaussian_predictor as gp
import classifier_pydl
from sklearn.metrics import accuracy_score, classification_report

def compare_data_with_pydl_classifier():
    data_set = dataset(dataset_name='iris')
    data_level_0 = data_set.data
    data_level_0.load_predictor('gaussian')
    data_level_0.generate_more_data()
    """
    data.add_gen_data(gen_data=new_samples,y_index=-1)
    x_train,x_test,y_train,y_test = data_obj.split_train_test(data.x_bin,data.y,test_size=0.2)
    clasfi = classifier_pydl.classify_with_default_error(
        x_bin=x_train,y=y_train,max_depth=4,min_sup=1,time=300)
    y_pred_test = clasfi.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy for real data only: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    ########################################
    #NOW LETS TRAIN ON compleetly FAKE DATA
    #########################################
    clasfi = classifier_pydl.classify_with_default_error(
        x_bin=data.x_gen_bin,y=data.y_gen,max_depth=4,min_sup=1,time=300)
    tree = clasfi.tree_
    l.VizTree(tree)
    y_pred_fake = clasfi.predict(data.x_bin)
    accuracy = accuracy_score(data.y, y_pred_fake)
    print(f"Accuracy for training on fake data alone: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(data.y, y_pred_fake))
    """




if __name__ == '__main__':
    compare_data_with_pydl_classifier()