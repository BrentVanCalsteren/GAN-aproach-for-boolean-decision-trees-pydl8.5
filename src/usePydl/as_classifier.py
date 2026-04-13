
from pydl85 import DL85Classifier
import helper
from sklearn.metrics import accuracy_score, classification_report



def train_clasfi(X,Y):
    clasfi = DL85Classifier(
        max_depth=6,  # maximum depth of the tree
        min_sup=1,  # minimum number of samples per leaf
        time_limit=30,  # 0 = no time limit (search for optimal tree)
        #verbose=True  # print progress
    )

    # Train the model
    clasfi.fit(X, Y)
    helper.VizTree(clasfi.tree_)
    return clasfi




def clasfi_test(clasfi,X_test,y_test):
    # Predict and evaluate
    y_pred = clasfi.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    complete_x, missing_x, dl_x, _,complete_y, missing_y,dl_y = helper.prep_data_for_pydl('heart_disease')
    #TODO: implement sample inbalance detection if needed -> not prio right now
    X_train, X_test, y_train, y_test = helper.randomize_data(dl_x, dl_y)
    clasfi = train_clasfi(X_train, y_train)
    clasfi_test(clasfi, X_test, y_test)