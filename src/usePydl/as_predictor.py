import random
import numpy as np
from pydl85 import DL85Predictor
from pyexpat import features
from scipy.stats import norm
import helper
from data_obj import dataset
import src.chanceCalc.chance_calc as chance
from sklearn.mixture import GaussianMixture


def gen_dl85predictor(data,error_fun,leaf_fun,max_depth,min_sup,time):
    predi = DL85Predictor(error_function=error_fun,leaf_value_function=leaf_fun,max_depth=max_depth,min_sup=min_sup, time_limit=time)
    predi.fit(data.x_bin)
    return predi


###############################
#########error functions############
###############################

def prob_norm_error(samples,p):
    def error(tids):
        sub_samples = np.array(samples[list(tids)])
        n = len(sub_samples)
        if n <= 1:
            return 1
        total_prob = 0
        for i in range(len(sub_samples)):
            sample = sub_samples[i]
            other = np.delete(sub_samples, i, axis=0)
            features = other.T
            distributions = chance.get_distr_for_features(features)
            sample_prob = chance.calc_likelihood_scores_samples(distributions,np.array([sample]))
            if np.min(sample_prob) > p:
                total_prob += np.min(sample_prob)
            else:
                total_prob -= np.min(sample_prob)
        return -total_prob
    return error

##TEST FUNCTION
def prob_norm_error2(samples):
    def error(tids):
        sub_samples = samples[list(tids)]
        if len(sub_samples) <= 1:
            return 0.0
        features = sub_samples.T
        total_neg_log_likelihood = 0.0
        for feat_vals in features:
            gm = GaussianMixture(n_components=1).fit(feat_vals.reshape(-1, 1))
            total_neg_log_likelihood -= gm.score_samples(feat_vals.reshape(-1, 1)).sum()
        return total_neg_log_likelihood
    return error


##########################"
#######leaf val##########
#########################
def leaf_value(samples):
    def value(tids):
        features = np.array(samples[list(tids)]).T
        print(features.shape)
        distributions = chance.get_distr_for_features(features)
        return distributions
    return value

def try_make_discriminator():
    max_depth = 4 #2**3 = 8 clusters
    max_bin_len_feat = 16
    num_features = 5
    data = dataset("iris", y_seperated=False,max_bin_len_feat=max_bin_len_feat,num_features=num_features)
    #error_fun = prob_norm_error(data.x, p=0.75)
    error_fun = prob_norm_error2(data.x) #test
    leaf_val = leaf_value(data.x)
    predictor = gen_dl85predictor(data,error_fun,leaf_val,max_depth,5,100)
    leafs = helper.get_all_leaves(predictor.tree_)
    distr_matrix = []
    errors = []
    for leaf in leafs:
        distr_matrix.append(leaf['value'])
        errors.append(leaf['error'])
    generate_new_data(distr_matrix,errors,n=10, num_features=num_features, conf_tresh=0.8)

def generate_new_data(distr_matrix,errors, n, num_features, conf_tresh=0.5):

    samples = []
    while len(samples) < n:
        z = np.random.random(size=(100, num_features))
        z_scores = np.array([0.0] * n)
        for i, distributions in enumerate(distr_matrix):
            z_prob_matrix = chance.calc_normalised_confidence(distributions, z)
            for j in range(n):
                if z_scores[j] < np.min(z_prob_matrix[j]):
                    z_scores[j] = np.min(z_prob_matrix[j])
        indices = np.argsort(z_scores)[-10:]
        candidates = z[indices]
        scores = z_scores[indices]
        for i, candidate in enumerate(candidates):
            if scores[i] > conf_tresh:
                samples.append((scores[i],candidate))
    print(f'found samples{samples}')





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




