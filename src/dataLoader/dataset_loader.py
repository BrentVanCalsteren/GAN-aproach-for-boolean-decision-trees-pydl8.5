import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional, Union, Tuple

class DatasetLoader:
    """Simplified loader for CSV/data files. Assumes first row is header."""

    MISSING_VAL_STRINGS = ['?', 'NA', 'N/A', 'null', 'NULL', 'None', '', ' ']

    def __init__(self, file_path: Union[str, Path],y_seperated = True):
        self.file_path = Path(file_path)
        self.dataframe: Optional[pd.DataFrame] = None
        self.y_seperated = y_seperated

        # All samples
        self.data_samples_X: Optional[np.ndarray] = None
        self.data_samples_Y: Optional[np.ndarray] = None

        # Complete vs. missing subsets
        self.complete_X: Optional[np.ndarray] = None
        self.complete_Y: Optional[np.ndarray] = None
        self.missing_X: Optional[np.ndarray] = None
        self.missing_Y: Optional[np.ndarray] = None

    def load(self, encoding: str = 'utf-8',y_ind:int = -1) -> None:
        """
        Load the CSV file. Assumes header is present and delimiter is auto-detected.
        The last column is treated as target (Y), all others as features (X).
        """
        try:
            # Let pandas detect delimiter automatically
            df = pd.read_csv(
                self.file_path,
                sep=None,                 # auto-detect delimiter
                header=0,                 # always use first row as header
                encoding=encoding,
                engine='python',
                on_bad_lines='skip'
            )
            self.dataframe = df

            # Convert to numpy array
            data_array = df.to_numpy()

            # Last column is Y, all others are X
            if data_array.shape[1] < 2:
                raise ValueError("Dataset must have at least two columns (features + target).")
            if self.y_seperated:
                X_raw = np.delete(data_array, y_ind, axis=1)
                Y_raw = data_array[:, y_ind]
            else:
                X_raw = data_array
                Y_raw = None

            # Identify rows with missing values in X (based on raw placeholders)
            missing_mask = self._get_missing_mask(X_raw)

            # Convert features (numeric -> float, else string)
            self.data_samples_X = self._convert_features(X_raw)
            self.data_samples_Y = Y_raw

            # Split into complete and missing subsets
            self.complete_X = self.data_samples_X[~missing_mask]
            self.missing_X = self.data_samples_X[missing_mask]
            self.complete_Y = Y_raw[~missing_mask] if Y_raw is not None else None
            self.missing_Y = Y_raw[missing_mask] if Y_raw is not None else None

            print(f"Loaded {self.data_samples_X.shape[0]} samples.")
            print(f"  Complete: {self.complete_X.shape[0]}")
            print(f"  Missing:  {self.missing_X.shape[0]}")

        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

    def _get_missing_mask(self, raw_X: np.ndarray) -> np.ndarray:
        """Return boolean mask for rows containing any missing placeholder."""
        mask = np.zeros(raw_X.shape[0], dtype=bool)
        for col in range(raw_X.shape[1]):
            col_data = raw_X[:, col].astype(str)
            mask |= np.isin(col_data, self.MISSING_VAL_STRINGS)
        return mask

    def _convert_features(self, samples: np.ndarray) -> np.ndarray:
        """
        Convert each column to float if mostly numeric, else keep as string.
        Missing placeholders become NaN (numeric) or '' (string).
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

    def get_y_all(self) -> np.ndarray:
        return self.data_samples_Y

    def get_y_complete(self) -> np.ndarray:
        return self.complete_Y

    def get_y_missing(self) -> np.ndarray:
        return self.missing_Y


def load_dataloader_by_name(dataset_name: str, main_dir: str = 'src',
                            data_subdir: str = 'datasets',y_seperated:bool=True, **kwargs) -> DatasetLoader:
    """
    Find and load a dataset by name from main_dir/data_subdir/dataset_name/dataset_name.csv (or .data).
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
            loader = DatasetLoader(candidate,y_seperated)
            loader.load(**kwargs)
            return loader

    raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {base_path}")



def standardize_to_num(arr):
    np_arr = np.asarray(arr)
    if np.issubdtype(np_arr.dtype, np.number):
        print("Already numeric.")
        return scale_array(np_arr)
    try:
        y_num = np_arr.astype(float)
        print("Converted numeric strings to float.")
        return scale_array(y_num)
    except (ValueError, TypeError):
        pass

    unique_values = np.unique(np_arr)
    print(f"Unique Y's: {unique_values}")
    value_to_idx = {val: i for i, val in enumerate(unique_values)}
    indices = np.array([value_to_idx[val] for val in np_arr])
    return scale_array(indices)


def standardize_2d_array(array):
    return np.array([standardize_to_num(x) for x in array])

def scale_array(num_array):
    arr = np.asarray(num_array, dtype=float)
    min_val = arr.min()
    max_val = arr.max()

    # Avoid division by zero when all values are equal
    if max_val - min_val == 0:
        return np.zeros_like(arr)

    return (arr - min_val) / (max_val - min_val)