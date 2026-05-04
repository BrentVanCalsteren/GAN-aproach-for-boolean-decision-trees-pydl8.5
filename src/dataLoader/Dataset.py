import parso.pgen2

import src.dataLoader.dataset_loader as loader
import numpy as np
from typing import List
from src.dataLoader.data import Data

class dataset:
    samples_complete = None
    samples_missing = None
    root_data = None
    active_splits = []



    def __init__(self,**kwargs):
        self.reload_data(**kwargs)


    def reload_data(self,dataset_name='iris',bin_length=-1):
        loaded_data = loader.load_dataloader_by_name(dataset_name)
        self.samples_complete = loaded_data.get_x_complete()
        self.samples_missing = loaded_data.get_x_missing()
        features_scaled = loader.standardize_2d_features(self.samples_complete.T)
        self.root_data = Data(feat_scaled=features_scaled,bin_length=bin_length)

    def split_data_on_features(self, n_splits=1, feature_splits=None):
        if feature_splits is None:
            feature_splits = []
        if self.root_data is None:
            print("No data loaded")
            return
        datas = [self.root_data]
        if feature_splits is None:
            print("No feature splits given, splitting on min unique feature")

        while n_splits > 0:
            new_datas = []
            for data in datas:
                data.split_data_on_index()
                new_datas.extend(data.child_datas)
            datas = new_datas
            n_splits -= 1

    def get_data_at_split_depth(self, depth):
        datas = []
        self.root_data.get_data_at_depth(datas, target_depth=depth)
        ids = [id(d) for d in datas]
        print(f"Unique nodes: {len(set(ids))}, Total: {len(ids)}")  # duplicates if different
        if not datas:
            print("Datas is empty")
        return datas

    def load_predictors(self,datas:List[Data], predictor_types: List[str],max_depth=3,time=100):
        if len(predictor_types) != len(datas):
            print("Number of predictors does not match number of datas")
        if not datas:
            print("Datas is empty")
        for i, data in enumerate(datas):
            data.load_predictor(predictor_types[i],max_depth=max_depth,time=time)

    def gen_new_samples_for_datalist(self, datas:List[Data], n=100, conf=0.8):
        n_max = len(self.root_data.x)
        if not n or n == -1: n = n_max
        for data in datas:
            n_data = len(data.x)
            n_split = int((n_data / n_max) * n)
            data.generate_more_data(n=n_split,conf=conf)
        self.gen_data_for_parents(datas)

    def gen_data_for_parents(self, datas: List[Data]):
        current_level = datas
        while current_level:
            parent_to_children = {}
            for child in current_level:
                if child.parent_data is not None:
                    parent_to_children.setdefault(child.parent_data, []).append(child)

            if not parent_to_children:
                break

            for parent, children in parent_to_children.items():
                parent.x_gen = np.vstack([child.x_gen for child in children])
                parent.set_bin_x_gen()

            current_level = list(parent_to_children.keys())









