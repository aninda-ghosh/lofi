from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from PIL import Image as im
import torch
from PIL import ImageFile
import glob
from tqdm import tqdm
import rasterio
import numpy as np

class LoFiDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        #read all the csv in the directory
        df_paths = glob.glob(dataset_path + "/*.csv")

        self.image_patches_count = 0

        self.harvest_image_paths = []
        self.planting_image_paths = []

        for df_path in tqdm(df_paths):
            df = pd.read_csv(df_path)
            for index, row in df.iterrows():
                self.harvest_image_paths.append(row['harvest_image_path'])
                self.planting_image_paths.append(row['planting_image_path'])
                self.image_patches_count += 1

    def __len__(self):
        return self.image_patches_count

    def __getitem__(self, index):
        with rasterio.open(self.harvest_image_paths[index]) as harvest_image:
            harvest_data = harvest_image.read()
            harvest_data = np.nan_to_num(harvest_data)  # Replace NaN with zero
            max_harvest = harvest_data.max()
            if max_harvest != 0:
                harvest_data = (harvest_data / max_harvest * 255).astype(np.uint8)

        with rasterio.open(self.planting_image_paths[index]) as planting_image:
            planting_data = planting_image.read()
            planting_data = np.nan_to_num(planting_data)  # Replace NaN with zero
            max_planting = planting_data.max()
            if max_planting != 0:
                planting_data = (planting_data / max_planting * 255).astype(np.uint8)

        image = np.stack([harvest_data, planting_data], axis=0)
        image = image.reshape(6, image.shape[2], image.shape[3]).astype(np.float32)
        pixel_values = torch.tensor(image, dtype=torch.float32) / 127.5 - 1.0
        return pixel_values