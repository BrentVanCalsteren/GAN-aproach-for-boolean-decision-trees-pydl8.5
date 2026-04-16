import src.dataLoader.data_obj as data_obj
import src.usePydl.leaf as l
import predictor.gaussian_predictor as gp
import classifier_pydl
from sklearn.metrics import accuracy_score, classification_report

def compare_data_with_pydl_classifier():
    data = data_obj.dataset( #first load data without seperating the y_target
        dataset_name='iris',
        num_features=10, #iris_dataset has only 5 feat (including y)
        max_bin_len_feat=10,
        y_seperated=False,
    )
    data.shuffle_data()
    predictor = gp.GaussianPredictor(data,max_depth=3,min_sup=1,time=100)
    predictor.generate_tree()
    new_samples = predictor.generate_new_data(conf_trash=0.8,number_of_new_samples=100)
    data.reload_data(
        dataset_name='iris',
        num_features=10,
        max_bin_len_feat=10,
        y_seperated=True, #reload but with seperating the y_target
        y_index=-1
    )
    data.shuffle_data()
    data.add_gen_data(gen_data=new_samples,y_index=-1)
    x_train,x_test,y_train,y_test = data_obj.split_train_test(data.x_bin,data.y,test_size=0.2)
    clasfi = classifier_pydl.classify_with_default_error(
        x_bin=x_train,y=y_train,max_depth=3,min_sup=1,time=100)
    y_pred = clasfi.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for real data only: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    ########################################
    #NOW LETS TRAIN ON compleetly FAKE DATA
    #########################################
    clasfi = classifier_pydl.classify_with_default_error(
        x_bin=data.x_gen_bin,y=data.y_gen,max_depth=3,min_sup=1,time=100)
    tree = clasfi.tree_
    l.VizTree(tree)
    y_pred = clasfi.predict(data.x_bin)
    y_real = data.y
    accuracy = accuracy_score(data.y, y_pred)
    print(f"Accuracy for training on fake data alone: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(data.y, y_pred))




if __name__ == '__main__':
    compare_data_with_pydl_classifier()