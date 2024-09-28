import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from geoclip.train.dataloader import GeoDataLoader

batch_size=1
ds = pd.read_csv("/home/rvl/Documents/rvl/pohsun/datasets/taipei.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = GeoDataLoader(ds, "") 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

points = set()
bar = tqdm(enumerate(train_loader), total=len(train_loader))

for _ ,(imgs_A, imgs_B, imgs_C, gps) in bar:
    x, y, theta = gps
    points.add((x.item(), y.item(), theta.item()))

print(len(points))

df = pd.DataFrame({
    'LAT': [f"{coordinate[0]:.6f}" for coordinate in points],
    'LON': [f"{coordinate[1]:.6f}" for coordinate in points],
    'HEAD': [f"{coordinate[2]:.6f}" for coordinate in points],
})

df.to_csv(f"/home/rvl/Documents/rvl/pohsun/datasets/taipei_coordinate_{len(df)}.csv", index=False)

