import math
import os
import PIL
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional
from PIL import Image
from src.data.encoders.encoder_loader import load_encoder

class DatasetLoader:
    MISSING_VAL_STRINGS = ['?', 'NA', 'N/A', 'null', 'NULL', 'None', '', ' ']
    RESOLUTION = (24, 24)
    LABEL_INDEX = -1
    MAX_IM_EACH_CLASS = 40
    CAP_NUM_FEATS = 40

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.encoder = None
        self.complete_X: Optional[np.ndarray] = None
        self.complete_Y: Optional[np.ndarray] = None
        self.missing_X: Optional[np.ndarray] = None
        self.missing_Y: Optional[np.ndarray] = None
        self.fix_image_args()

    def fix_image_args(self):
        sqrt_out = math.ceil(math.sqrt(self.CAP_NUM_FEATS))
        self.CAP_NUM_FEATS = sqrt_out ** 2
        if not self.RESOLUTION[0] == self.RESOLUTION[1]:
            self.RESOLUTION = (self.RESOLUTION[0], self.RESOLUTION[0])

    def load_tabular_data(self):
        try:
            df = pd.read_csv(
                self.file_path,
                sep=None,
                header=0,
                encoding='utf-8',
                engine='python',
                on_bad_lines='skip'
            )
            raw_samples = df.to_numpy()
            missing_mask = self._get_missing_mask(raw_samples)
            self.complete_X = raw_samples[~missing_mask]
            self.missing_X = raw_samples[missing_mask]

            print(f"Loaded {raw_samples.shape[0]} samples: complete {self.complete_X.shape[0]}, missing {self.missing_X.shape[0]}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

    def _get_missing_mask(self, data: np.ndarray) -> np.ndarray:
        mask = np.zeros(data.shape[0], dtype=bool)
        for col in range(data.shape[1]):
            col_data = data[:, col].astype(str)
            mask |= np.isin(col_data, self.MISSING_VAL_STRINGS)
        return mask

    def load_image_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Directory {self.file_path} does not exist.")
        features = []
        labels = []
        class_folders = [f for f in self.file_path.iterdir() if f.is_dir()]
        class_folders.sort()

        class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}
        print(f"Found classes: {class_to_idx}")

        def check_greyscale(img: PIL.Image.Image):
            if img.mode in ('L', '1'):
                return True

        greyscale = False
        for folder in class_folders:
            label_idx = class_to_idx[folder.name]
            for i, img_file in enumerate(folder.iterdir()):
                if i > self.MAX_IM_EACH_CLASS:
                    break
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    try:
                        img = Image.open(img_file)
                        if greyscale or check_greyscale(img):
                            greyscale = True
                            img = img.convert('L')
                        else:
                            img = img.convert('RGB')

                        if self.RESOLUTION:
                            img = img.resize(self.RESOLUTION)

                        img_array = np.asarray(img, dtype=np.float32)
                        features.append(img_array.flatten())
                        labels.append(label_idx)
                    except Exception as e:
                        print(f"Failed to load image {img_file}: {e}")

        if not features:
            raise ValueError(f"No valid images found in {self.file_path}")

        features = np.array(features)
        labels = np.array(labels).reshape(-1, 1)

        if features.shape[1] > self.CAP_NUM_FEATS:
            reduced_features = self.reduce_features(features, greyscale)
            self.complete_X = np.hstack((reduced_features, labels))
        else:
            self.complete_X = np.hstack((features, labels))

    def reduce_features(self, features, greyscale):
        print("Reducing features...")
        self.encoder = load_encoder(samples=features, type='pca', output_dim=self.CAP_NUM_FEATS)
        reduced_features = self.encoder.transform(features)
        return reduced_features

def load_dataloader_by_name(dataset_name: str, main_dir: str = 'GAN-aproach-for-boolean-decision-trees-pydl8.5',
                            data_subdir: str = 'datasets', data_type='tabular') -> DatasetLoader:
    file_path = Path().resolve()
    str_path = str(file_path)
    index = str_path.find(main_dir)
    if index == -1:
        raise ValueError(f"Main directory '{main_dir}' not found in path: {str_path}")
    main_path = Path(str_path[:index + len(main_dir)])
    base_path = main_path / data_subdir
    if data_type == 'tabular':
        for ext in ['.csv', '.data']:
            candidate = base_path / dataset_name / f"{dataset_name}{ext}"
            if candidate.exists():
                print(f"Loading dataset from: {candidate}")
                loader = DatasetLoader(candidate)
                loader.load_tabular_data()
                return loader
    elif data_type == 'image':
            folder = base_path / dataset_name
            loader = DatasetLoader(folder)
            loader.load_image_data()
            return loader
    raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {base_path}")

def array_to_image(arr: np.ndarray):
    if arr.ndim == 1:
        total_elements = arr.shape[0]
        side_length = math.ceil(math.sqrt(total_elements))
        target_size = side_length * side_length
        padding_needed = target_size - total_elements

        if padding_needed > 0:
            arr = np.pad(arr, (0, padding_needed), mode='constant', constant_values=0)
        arr = arr.reshape((side_length, side_length))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_image_to_folder(img, folder_name="output", filename="generated_image.png"):
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, filename)
    img.save(file_path)
    print(f"Success! Image saved to: {file_path}")
