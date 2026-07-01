import os
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional

MISSING_VAL_STRINGS = ['?', 'NA', 'N/A', 'null', 'NULL', 'None', '', ' ']


class TabularData:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.max_samples = 1000
        self.complete_X: Optional[np.ndarray] = None
        self.missing_X: Optional[np.ndarray] = None

    def load_tabular_data(self):
        try:
            df = pd.read_csv(
                self.file_path,
                sep=None,
                header=0,
                encoding='utf-8',
                engine='python',
                on_bad_lines='skip',
                nrows=self.max_samples,
            )
            raw_samples = df.to_numpy()
            missing_mask = _get_missing_mask(raw_samples)
            self.complete_X = raw_samples[~missing_mask]
            self.missing_X = raw_samples[missing_mask]

            print(f"Loaded {raw_samples.shape[0]} samples: complete {self.complete_X.shape[0]}, missing {self.missing_X.shape[0]}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

    def save_output_to_folder(self, tabular_data: np.ndarray, labels: np.ndarray, folder_name="output", filename="output.cvs"):
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, filename)
        np.savetxt(file_path, tabular_data, delimiter=",", fmt="%s")
        print(f"Successfully saved data to: {file_path}")

    def get_samples(self):
        if self.complete_X is None:
            raise None
        return self.complete_X[:,:-1], self.complete_X[:,-1].reshape(-1, 1)


def _get_missing_mask(data: np.ndarray) -> np.ndarray:
    mask = np.zeros(data.shape[0], dtype=bool)
    for col in range(data.shape[1]):
        col_data = data[:, col].astype(str)
        mask |= np.isin(col_data, MISSING_VAL_STRINGS)
    return mask