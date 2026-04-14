import random
import numpy as np
from pydl85 import DL85Predictor
from pyexpat import features
from scipy.stats import norm
import helper
from data_obj import dataset
import src.chanceCalc.chance_calc as chance


def gen_dl85predictor(data,error_fun,leaf_fun,max_depth,min_sup,time):
    predi = DL85Predictor(error_function=error_fun,leaf_value_function=leaf_fun,max_depth=max_depth,min_sup=min_sup, time_limit=time)
    predi.fit(data.x_bin)
    return predi


def try_make_discriminator():
    max_depth = 2
    max_bin_len_feat = 10
    num_features = 4
    data = dataset("iris", y_seperated=False,max_bin_len_feat=max_bin_len_feat,num_features=num_features)
    error_fun = prob_norm_error(data.x)
    leaf_val = leaf_value(data.x)
    predictor = gen_dl85predictor(data,error_fun,leaf_val,max_depth,2,300)
    leafs = helper.get_all_leaves(predictor.tree_)
    print(leafs)
    helper.VizTree(predictor.tree_)


###############################
#########error functions############
###############################

def prob_norm_error(samples):
    def error(tids):
        sub_samples = np.array(samples[list(tids)])
        n = len(sub_samples)
        if n <= 1:
            return 0.0
        features = sub_samples.T
        distributions = chance.get_distr_for_features(features, "norm")
        #this is not completly correct since you should remove the sample everytime out the distr
        log_likelyhood = chance.calc_likelihood_scores_samples(distributions,samples)
        return log_likelyhood
    return error

##########################"
#######leaf val##########
#########################
def leaf_value(samples):
    def value(tids):
        features = np.array(samples[list(tids)]).T
        distributions = chance.get_distr_for_features(features, "norm")
        return distributions
    return value

if __name__ == "__main__":
    try_make_discriminator()


"""
n_samples = 100
#creat random numerical noise
_, n_features = scaled_x.shape
z = np.random.random(size=(n_samples, n_features))
z_bin = helper.convert_num_specific_bin_length(z.T,clusters)

#test  replicate original data
original_data = scaled_x

def error_for_building_tree(tids):
    a = np.array(z_bin[list(tids)])
    pred_scores = np.array([confidence(scaled_x[i], x) for i, x in enumerate(disc.predict(a))])
    return float(pred_scores.mean())

def leaf_val_gen(tids):
    a = z[list(tids)]
    return a

# --- 4. Train the generator ---
generator = DL85Predictor(
    max_depth=5,
    min_sup=1,
    error_function=error_for_building_tree,
    leaf_value_function=leaf_val_gen,
    time_limit=300
)

generator.fit(z_bin)
"""




