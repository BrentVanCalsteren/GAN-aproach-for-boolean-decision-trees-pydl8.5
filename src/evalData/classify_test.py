#train on fake, test on real
from pydl85 import DL85Classifier
from src.usePydl.generate_data import generate_new_data
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, classification_report
from src.evalData.discriminator import DLDiscriminator, NNDiscriminator
from src.evalData.sanity_checks import *
from src.evalData.statistic_sim import *
import warnings
warnings.filterwarnings("ignore")

def eval_data():
    x, x_bin, x_gen, x_gen_bin, y, y_gen = generate_new_data(
        pred_type="ensemble",
        dataset_name='iris',
        try_splits=0,
        y_index = None,
        time=300,
        conf=0.95,
        n_samples=-1
    )
    print("CHECKS ON NORMALIZED DATA")
    do_all_sanity_checks(x,x_gen)
    do_all_statistic_tests(x,x_gen)
    print("CHECKS ON BINARY BINNED DATA")
    do_all_sanity_checks(x_bin, x_gen_bin)
    do_all_statistic_tests(x_bin, x_gen_bin)
    train_on_gen_test_on_real()
    train_discriminators()


def train_on_gen_test_on_real():
    x, x_bin, x_gen, x_gen_bin, y, y_gen = generate_new_data(
        pred_type="ensemble",
        dataset_name='iris',
        try_splits=0,
        y_index = -1,
        time=300,
        conf=0.9,
        n_samples=-1
    )
    print("result on real data")
    x_train, x_test, y_train, y_test = split_train_test(x_bin, y, test_size=0.2)
    classify_test_pydl(x_train, x_test, y_train, y_test)
    print("result on gen data")
    classify_test_pydl(x_gen_bin, x_bin, y_gen, y)

def train_discriminators():
    x, x_bin, x_gen, x_gen_bin, y, y_gen = generate_new_data(
        pred_type="ensemble",
        dataset_name='iris',
        try_splits=0,
        y_index = None,
        time=300,
        conf=0.95,
        n_samples=-1
    )
    d = DLDiscriminator(x_bin,x_gen_bin)
    print("check discriminator DL result data")
    d.classify()
    d = NNDiscriminator()
    d.fit(x,x_gen)
    y_true,y_pred = d.score(x,x_gen)
    print("discriminator NN result data")
    print("\nClassification Report nn:")
    print(classification_report(y_true, y_pred))




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

def create_classifier_default(x_bin,y,max_depth=3,min_sup=2,time=100):
    clasfi = DL85Classifier(max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(x_bin, y)
    return clasfi

if __name__ == "__main__":
    eval_data()