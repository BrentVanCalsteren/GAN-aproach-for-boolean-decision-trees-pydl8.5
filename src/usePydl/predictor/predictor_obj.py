from pydl85 import DL85Predictor

class Predictor:
    def __init__(self,samples_bin,error_fun,leaf_val,max_depth,min_sup,time):
        self.samples_bin = samples_bin
        self.predictor = DL85Predictor(error_function=error_fun,leaf_value_function=leaf_val,
                                       max_depth=max_depth,min_sup=min_sup, time_limit=time)
    def generate_tree(self):
        self.predictor.fit(self.samples_bin)

    def load_new_data(self,samples_bin):
        self.samples_bin = samples_bin
        self.generate_tree()

    def predict(self,samples_bin):
        return self.predictor.predict(samples_bin)

    def generate_new_data(self):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def calc_norm_conf_each_sample(self, distributions, samples):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def get_distr(self, feature_array):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')

    def get_distributions(self, features):
        distr_funs = []
        for feat in features:
            distr_funs.append(self.get_distr(feat))
        return distr_funs  # [dstr-f1, dstr-f2, dstr-f3,...]

    def get_error_sample(self, distributions, samples):
        error = 1 - self.calc_norm_conf_each_sample(distributions, samples)
        return error




