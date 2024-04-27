from torch.utils.data import Dataset
import pandas as pd
import geopandas as gpd
import torch
import rasterio
import numpy as np
import cv2

class LoFiInstanceDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        #read all the csv in the directory
        df = pd.read_csv(dataset_path)
        
        self.image_patches_count = 0

        self.harvest_image_paths = []
        self.planting_image_paths = []
        self.masks_paths = []

        for index, row in df.iterrows():
            self.harvest_image_paths.append(row['harvest_image_path'])
            self.planting_image_paths.append(row['planting_image_path'])
            self.masks_paths.append(row['mask_path'])
            self.image_patches_count += 1

    def __len__(self):
        return self.image_patches_count

    def __getitem__(self, index):
        geographical_bounds = None

        with rasterio.open(self.harvest_image_paths[index]) as harvest_image:
            harvest_data = harvest_image.read()
            geographical_bounds = harvest_image.bounds
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


        # Read geojson file
        masks = gpd.read_file(self.masks_paths[index])['geometry']

        im_masks, im_bounding_boxes, im_labels = self.convert_to_pixel_coordinates(geographical_bounds, (pixel_values.shape[1], pixel_values.shape[2]), masks)
        im_masks = torch.tensor(im_masks, dtype=torch.float32)
        im_bounding_boxes = torch.tensor(im_bounding_boxes, dtype=torch.float32)
        im_labels = torch.tensor(im_labels, dtype=torch.int64)

        target = {
            "boxes": im_bounding_boxes,
            "labels": im_labels,
            "masks": im_masks
        }

        return pixel_values, target
    
    def convert_to_pixel_coordinates(self, geographical_bounds, image_bounds, masks):
        left, bottom, right, top = geographical_bounds
        image_width, image_height = image_bounds
        
        # Pre-calculate constants
        width_ratio = image_width / (right - left)
        height_ratio = image_height / (top - bottom)

        instance_masks = []
        bounding_boxes = []
        labels = []
        for mask in masks:
            mask_background = np.zeros((image_height, image_width), dtype=np.uint8)
            mask_image_coords = np.array(mask.exterior.coords)
            mask_image_coords[:, 0] = (mask_image_coords[:, 0] - left) * width_ratio
            mask_image_coords[:, 1] = (mask_image_coords[:, 1] - bottom) * height_ratio

            #cv2.fillPoly(mask_background, [mask_image_coords.astype(np.int32)], 255)
            #Use fillPoly to fill the mask with either 0 or 1
            cv2.fillPoly(mask_background, [mask_image_coords.astype(np.int32)], 1)

            # Append a copy of mask_background to instance_masks
            instance_masks.append(np.flipud(mask_background.copy()))
            bounding_boxes.append([mask_image_coords[:, 0].min(), mask_image_coords[:, 1].min(), mask_image_coords[:, 0].max(), mask_image_coords[:, 1].max()])
            labels.append(1)

        return np.array(instance_masks), np.array(bounding_boxes), np.array(labels)