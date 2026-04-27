#train on fake, test on real
from pydl85 import DL85Classifier
from src.usePydl.generate_data import generate_new_data
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, classification_report

def create_classifier_default(x_bin,y,max_depth=3,min_sup=2,time=100):
    clasfi = DL85Classifier(max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(x_bin, y)
    return clasfi

def eval_data():
    x, x_bin, x_gen, x_gen_bin, y, y_gen = generate_new_data(pred_type="gaussian_1D",dataset_name='bank',try_splits=2,y_index = -1)
    print("result on real data")
    x_train, x_test, y_train, y_test = split_train_test(x_bin, y, test_size=0.2)
    classify_test_pydl(x_train, x_test, y_train, y_test)
    print("result on gen data")
    classify_test_pydl(x_gen_bin, x_bin, y_gen, y)


def split_train_test(x, y, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=random.randint(1, 100))
    return x_train, x_test, y_train, y_test

def classify_test_pydl(x_train, x_test, y_train, y_test):
    clasfi = create_classifier_default(
        x_bin=x_train, y=y_train, max_depth=4, min_sup=1, time=300)
    y_pred_test = clasfi.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

if __name__ == "__main__":
    eval_data()