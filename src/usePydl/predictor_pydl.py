import numpy as np
from pydl85 import DL85Predictor
import src.chanceCalc.prob_estimator as probEstimator
from error_fun import prob_norm_error2
import leaf as l
from src.dataLoader.data_obj import dataset


########################
#Predicitors
#################"

def gen_dl85predictor(data,error_fun,leaf_fun,max_depth,min_sup,time):
    predi = DL85Predictor(error_function=error_fun,leaf_value_function=leaf_fun,max_depth=max_depth,min_sup=min_sup, time_limit=time)
    predi.fit(data.x_bin)
    return predi


################
#try make discriminator with predictor
##############


def try_make_discriminator():
    max_depth = 4 #2**4 = 8 clusters
    max_bin_len_feat = 16
    num_features = 5
    data = dataset("iris", y_seperated=False,max_bin_len_feat=max_bin_len_feat,num_features=num_features)
    #error_fun = prob_norm_error(data.x, p=0.75)
    error_fun = prob_norm_error2(data.x) #test
    leaf_val = l.leaf_val_gaussian_distributions(data.x)
    predictor = gen_dl85predictor(data,error_fun,leaf_val,max_depth,5,100)
    leafs = l.get_all_leaves(predictor.tree_)
    distrbutions_leafs = []
    errors = []
    for leaf in leafs:
        distrbutions_leafs.append(leaf['value'])
        errors.append(leaf['error'])
    generate_new_data(distrbutions_leafs,errors,n=10, num_features=num_features, conf_tresh=0.8)

####################
##Try generating data with dicriptor resutls
###################

def generate_new_data(distrbutions_leafs, n, num_features, conf_tresh=0.5,shifted=True):

    samples = []
    while len(samples) < n:
        z = np.random.random(size=(100, num_features))
        if shifted:
            z+=1
        z_scores = np.array([0.0] * n)
        for i, distributions in enumerate(distrbutions_leafs):
            z_prob_matrix = probEstimator.calc_normalised_confidence_gaussian(distributions, z)
            for j in range(n):
                if z_scores[j] < np.min(z_prob_matrix[j]):
                    z_scores[j] = np.min(z_prob_matrix[j])
        indices = np.argsort(z_scores)[-10:]
        candidates = z[indices]
        scores = z_scores[indices]
        for i, candidate in enumerate(candidates):
            if scores[i] > conf_tresh:
                samples.append((scores[i],candidate))
                print(f'found sample with score{scores[i]}')
    print(f'found samples{samples}')





if __name__ == "__main__":
    try_make_discriminator()




