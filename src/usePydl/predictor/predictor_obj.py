from pydl85 import DL85Predictor
from src.usePydl.leaf import get_all_leaves

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

    def get_leaf_vals(self):
        return get_all_leaves(self.predictor.tree_)

    def generate_new_data(self):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')