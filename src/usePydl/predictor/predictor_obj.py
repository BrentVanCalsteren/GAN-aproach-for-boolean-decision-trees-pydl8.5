from pydl85 import DL85Predictor
from src.usePydl.leaf import get_all_leaves

class Predictor:
    def __init__(self,data,error_fun,leaf_val,max_depth,min_sup,time):
        self.data = data
        self.predictor = DL85Predictor(error_function=error_fun,leaf_value_function=leaf_val,
                                       max_depth=max_depth,min_sup=min_sup, time_limit=time)
    def generate_tree(self):
        self.predictor.fit(self.data.x_bin)

    def load_new_data(self,data):
        self.data = data
        self.generate_tree()

    def predict(self,bin_data):
        return self.predictor.predict(bin_data)

    def get_leaf_vals(self):
        return get_all_leaves(self.predictor.tree_)

    def generate_new_data(self):
        raise NotImplementedError('if you wanna use this method you should implement it in child class')