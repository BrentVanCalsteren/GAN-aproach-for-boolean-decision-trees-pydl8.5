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
            #TODO: implement feature split
        while n_splits > 0:
            new_datas  = []
            for data in datas:
                data.split_data_on_index()
                data.get_data_at_depth(new_datas,1)
            datas = new_datas
            n_splits -=1

    def get_data_at_split_depth(self,depth):
        datas = []
        self.root_data.get_data_at_depth(datas,depth)
        return datas

    def load_predictors(self,datas:List[Data], predictor_types: List[str]):
        if len(predictor_types) != len(datas):
            print("Number of predictors does not match number of datas")
        for i, data in enumerate(datas):
            data.load_predictor(predictor_types[i])

    def generate_new_data(self,datas:List[Data],n=100,conf=0.8):
        for data in datas:
            data.generate_more_data(n=n,conf=conf)

    def combine_gen_data(self):
        pass









