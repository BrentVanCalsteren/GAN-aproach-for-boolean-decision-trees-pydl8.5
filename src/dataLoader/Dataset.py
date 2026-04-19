import src.dataLoader.dataset_loader as loader
import numpy as np
from src.dataLoader.data import Data

class dataset:
    samples_complete = None
    samples_missing = None
    data = None


    def __init__(self,**kwargs):
        self.reload_data(**kwargs)


    def reload_data(self,dataset_name='iris',bin_length=-1):
        loaded_data = loader.load_dataloader_by_name(dataset_name)
        self.samples_complete = loaded_data.get_x_complete()
        self.samples_missing = loaded_data.get_x_missing()
        features_scaled = loader.standardize_2d_features(self.samples_complete.T)
        self.data = Data(feat_scaled=features_scaled,bin_length=bin_length)










