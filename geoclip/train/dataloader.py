import os
import torch
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from PIL import Image as im
from torchvision import transforms
from torch.utils.data import Dataset

def img_train_transform():
    train_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

def img_val_transform():
    val_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return val_transform_list    


class GeoDataLoader(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.images_A, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        # try:
        #     dataset_info = pd.read_csv(dataset_file)
        # except Exception as e:
        #     raise IOError(f"Error reading {dataset_file}: {e}")

        dataset_info = dataset_file
        images_A = []
        images_B = []
        images_C = []
        coordinates = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            file_A = os.path.join(self.dataset_folder, row['IMG_FILE_A'])
            file_B = os.path.join(self.dataset_folder, row['IMG_FILE_B'])
            file_C = os.path.join(self.dataset_folder, row['IMG_FILE_C'])
            if exists(file_A) and exists(file_B) and exists(file_C):
                images_A.append(file_A)
                images_B.append(file_B)
                images_C.append(file_C)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                heading = float(row['head'])
                coordinates.append((latitude, longitude, heading))

        return images_A, coordinates

    def __len__(self):
        return len(self.images_A)

    def __getitem__(self, idx):
        img_path_A = self.images_A[idx]
        gps = self.coordinates[idx]

        image_A = im.open(img_path_A).convert('RGB')
        
        if self.transform:
            image_A = self.transform(image_A)
        else:
            trans = transforms.Compose([
                transforms.PILToTensor(),
            ])
            image_A = trans(image_A)

        return image_A, gps
