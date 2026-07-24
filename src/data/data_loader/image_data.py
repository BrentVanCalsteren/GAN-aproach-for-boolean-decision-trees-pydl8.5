import os
from pathlib import Path
import numpy as np
from PIL import Image
import random


valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
class ImageData:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.n_max_each_class = 50
        self.X = None
        self.Y = None
        self.n_dim = 1
        self.img_shape = None
        self.n_channels = 1
        self.samples = None


    def load_image_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Directory {self.file_path} does not exist.")

        features = []
        labels = []
        class_folders = [f for f in self.file_path.iterdir() if f.is_dir()]
        class_folders.sort()
        class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}
        print(f"Found classes: {class_to_idx}")

        is_greyscale = False #not a good solution when there are colored and grey images
        for folder in class_folders:
            image_files = [f for f in folder.iterdir() if f.suffix.lower() in valid_extensions]
            if self.n_max_each_class is not None and len(image_files) > self.n_max_each_class:
                image_files = random.sample(image_files, self.n_max_each_class)
            label_idx = class_to_idx[folder.name]
            for img_file in image_files:
                try:
                    img = Image.open(img_file)

                    if self.img_shape is None:
                        is_greyscale = img.mode in ('L', '1')
                        img = img.convert('L') if is_greyscale else img.convert('RGB')
                        img_array = np.asarray(img, dtype=np.float32)

                        self.img_shape = img_array.shape  # (H, W) or (H, W, 3)
                        self.n_channels = 1 if is_greyscale else 3
                    else:
                        img = img.convert('L') if is_greyscale else img.convert('RGB')
                        target_size = (self.img_shape[1], self.img_shape[0])  # (Width, Height)
                        if img.size != target_size:
                            img = img.resize(target_size)
                        img_array = np.asarray(img, dtype=np.float32)

                    features.append(img_array)
                    labels.append(label_idx)

                except Exception as e:
                    print(f"Failed to load image {img_file}: {e}")

        if not features:
            raise ValueError(f"No valid images found in {self.file_path}")

        self.X = np.array(features)
        self.Y = np.array(labels).reshape(-1, 1)

    def image_arr_to_tabular(self) -> np.ndarray:
        if self.X is None:
            raise ValueError("Data not loaded. Call load_image_data() first.")

        tabular_data = self.X.reshape(self.X.shape[0], -1)
        return tabular_data

    def tabular_to_image_arr(self, tabular: np.ndarray):
        if self.img_shape is None:
            raise ValueError("Original image shape is unknown. Load data first.")

        im_arr = tabular.reshape(tabular.shape[0], self.img_shape[0],self.img_shape[1])
        return im_arr

    def save_output_to_folder(self, tabular_data: np.ndarray, labels: np.ndarray,folder_name="output", filename="generated_image"):
        im_data = self.tabular_to_image_arr(tabular_data)

        for i,label in enumerate(labels):
            img_uint8 = np.clip(im_data[i], 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, filename + "_" +str(label) + ".png")
            img_pil.save(file_path)
            print(f"Success! Image saved to: {file_path}")

    def get_samples(self):
        return self.image_arr_to_tabular(), self.Y