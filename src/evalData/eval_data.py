#file is out of date

#train on fake, test on real
from pydl85 import DL85Classifier
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, classification_report
from src.evalData.discriminator import DLDiscriminator, NNDiscriminator
from src.evalData.sanity_checks import *
from src.evalData.statistic_sim import *
import warnings
warnings.filterwarnings("ignore")

def eval_data():
    #have to rewrite this since the way i load and store data has been changed
    pass


def train_on_gen_test_on_real(x,x_bin,y,x_gen,x_gen_bin,y_gen):
    print("result on real data")
    x_train, x_test, y_train, y_test = split_train_test(x_bin, y, test_size=0.2)
    classify_test_pydl(x_train, x_test, y_train, y_test)
    print("result on gen data")
    classify_test_pydl(x_gen_bin, x_bin, y_gen, y)

def train_discriminators(x,x_bin,x_gen,x_gen_bin):
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

if __name__ == "__main__":
    eval_data()