import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

MISSING_VAL_STRINGS = ['?', 'NA', 'N/A', 'null', 'NULL', 'None', '', ' ']


class TabularData:

  def __init__(self, file_path: str | Path, chunk_size: int = 200, seed: Optional[int] = 42):

    self.file_path = Path(file_path)
    self.chunk_size = chunk_size
    self.seed = seed
    self.complete_X: Optional[np.ndarray] = None
    self.missing_X: Optional[np.ndarray] = None
    self.chunk_files: List[Path] = []
    self.num_chunks: int = 0
    self.split_if_needed()

  def split_if_needed(self) -> List[Path]:
    if not self.file_path.exists():
      raise FileNotFoundError(f'File not found: {self.file_path}')

    df = pd.read_csv(
        self.file_path,
        sep=None,
        header=0,
        encoding='utf-8',
        engine='python',
        on_bad_lines='skip',
    )

    total_samples = len(df)
    if total_samples <= self.chunk_size:
      self.chunk_files = [self.file_path]
      self.num_chunks = 1
      return self.chunk_files

    # Randomly shuffle rows for unique chunking
    df_shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    parent_dir = self.file_path.parent
    stem = self.file_path.stem
    ext = self.file_path.suffix or '.csv'

    self.chunk_files = []
    self.num_chunks = int(np.ceil(total_samples / self.chunk_size))

    for i in range(self.num_chunks):
      chunk_df = df_shuffled.iloc[i * self.chunk_size : (i + 1) * self.chunk_size]
      chunk_path = parent_dir / f'{stem}_chunk_{i + 1}{ext}'
      chunk_df.to_csv(chunk_path, index=False)
      self.chunk_files.append(chunk_path)

    print(
        f'Dataset has {total_samples} samples. Split into {self.num_chunks}'
        f' chunks in: {parent_dir}')
    return self.chunk_files

  def loading_chunk(self, chunk_num: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    if chunk_num < 1 or chunk_num > self.num_chunks:
      raise IndexError(
          f'chunk_num {chunk_num} out of range. Valid chunks: 1 to'
          f' {self.num_chunks}'
      )

    target_file = self.chunk_files[chunk_num - 1]
    try:
      df = pd.read_csv(
          target_file,
          sep=None,
          header=0,
          encoding='utf-8',
          engine='python',
          on_bad_lines='skip',
      )
      raw_samples = df.to_numpy()
      missing_mask = _get_missing_mask(raw_samples)
      self.complete_X = raw_samples[~missing_mask]
      self.missing_X = raw_samples[missing_mask]

      print(
          f'Loaded chunk {chunk_num}/{self.num_chunks} ({target_file.name}): '
          f'complete {self.complete_X.shape[0]}, missing'
          f' {self.missing_X.shape[0]}'
      )

      return self.get_samples()

    except Exception as e:
      raise RuntimeError(f'Error loading dataset chunk from {target_file}: {e}')

  def save_output_to_folder(self, tabular_data: np.ndarray, labels: np.ndarray, folder_name='output', filename='output.csv',):
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, filename)

    if labels is not None:
      data_to_save = np.hstack([tabular_data, labels.reshape(-1, 1)])
    else:
      data_to_save = tabular_data

    np.savetxt(file_path, data_to_save, delimiter=',', fmt='%s')
    print(f'Successfully saved data to: {file_path}')

  def get_samples(self):
    if self.complete_X is None:
      raise ValueError("No data loaded yet. Call 'loading_chunk()' first.")
    return self.complete_X[:, :-1], make_num(self.complete_X[:, -1].flatten()).reshape(-1, 1)


def _get_missing_mask(data: np.ndarray) -> np.ndarray:
  mask = np.zeros(data.shape[0], dtype=bool)
  for col in range(data.shape[1]):
    col_data = data[:, col].astype(str)
    mask |= np.isin(col_data, MISSING_VAL_STRINGS)
  return mask

def make_num(raw_feature_data):
    try:
        num_arr = np.asarray(raw_feature_data, dtype=float)
    except (ValueError, TypeError):
        unique_values = np.unique(raw_feature_data)
        indexes = {val: idx for idx, val in enumerate(unique_values)}
        num_arr = np.array([indexes[val] for val in raw_feature_data])
        #maybe store the original strings? but would never need them (always want it to be numbers)
    return num_arr