import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional

class DatasetLoader:
    """Simplified loader for CSV/data files. Assumes first row is header."""

    MISSING_VAL_STRINGS = ['?', 'NA', 'N/A', 'null', 'NULL', 'None', '', ' ']

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.dataframe: Optional[pd.DataFrame] = None

        # All samples
        self.data_samples_X: Optional[np.ndarray] = None
        self.data_samples_Y: Optional[np.ndarray] = None

        # Complete vs. missing subsets
        self.complete_X: Optional[np.ndarray] = None
        self.complete_Y: Optional[np.ndarray] = None
        self.missing_X: Optional[np.ndarray] = None
        self.missing_Y: Optional[np.ndarray] = None

    def load(self, encoding: str = 'utf-8') -> None:
        """
        Load the CSV file with pandas, split samples into complete/missing
        """
        try:
            # Let pandas detect delimiter automatically
            df = pd.read_csv(
                self.file_path,
                sep=None,
                header=0,
                encoding=encoding,
                engine='python',
                on_bad_lines='skip'
            )
            self.dataframe = df

            data_array = df.to_numpy()#numpy array for missing mask
            raw_samples = data_array
            self.data_samples_X = self._convert_features(raw_samples)

            missing_mask = self._get_missing_mask(raw_samples)
            self.complete_X = self.data_samples_X[~missing_mask]
            self.missing_X = self.data_samples_X[missing_mask]

            print(f"Loaded {self.data_samples_X.shape[0]} "
                  f"samples: complete {self.complete_X.shape[0]},missing {self.missing_X.shape[0]}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

    def _get_missing_mask(self, data: np.ndarray) -> np.ndarray:
        """Return boolean mask for rows containing any missing placeholder."""
        mask = np.zeros(data.shape[0], dtype=bool)
        for col in range(data.shape[1]):
            col_data = data[:, col].astype(str)
            mask |= np.isin(col_data, self.MISSING_VAL_STRINGS)
        return mask

    def _convert_features(self, samples: np.ndarray) -> np.ndarray:
        """
        try converting everything to a float
        """
        converted = samples.copy().astype(object)
        for i in range(samples.shape[1]):
            col = samples[:, i]
            # Replace missing placeholders with NaN
            col_clean = np.where(np.isin(col, self.MISSING_VAL_STRINGS), np.nan, col)
            # Try numeric conversion
            num_col = pd.to_numeric(col_clean, errors='coerce')
            if (~pd.isna(num_col)).sum() / len(num_col) >= 0.5:
                converted[:, i] = num_col
            else:
                converted[:, i] = np.where(np.isin(col, self.MISSING_VAL_STRINGS), '', col.astype(str))
        return converted

    # ---------- Getters ----------
    def get_x_all(self) -> np.ndarray:
        return self.data_samples_X

    def get_x_complete(self) -> np.ndarray:
        return self.complete_X

    def get_x_missing(self) -> np.ndarray:
        return self.missing_X


def load_dataloader_by_name(dataset_name: str, main_dir: str = 'src',
                            data_subdir: str = 'datasets', **kwargs) -> DatasetLoader:
    """
    Find and load a dataset by name from main_dir/data_subdir/dataset_name/dataset_name.csv/.data
    """
    file_path = Path().resolve()
    str_path = str(file_path)
    index = str_path.find(main_dir)
    if index == -1:
        raise ValueError(f"Main directory '{main_dir}' not found in path: {str_path}")
    main_path = Path(str_path[:index + len(main_dir)])
    base_path = main_path / data_subdir

    for ext in ['.csv', '.data']:
        candidate = base_path / dataset_name / f"{dataset_name}{ext}"
        if candidate.exists():
            print(f"Loading dataset from: {candidate}")
            loader = DatasetLoader(candidate)
            loader.load(**kwargs)
            return loader

    raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {base_path}")