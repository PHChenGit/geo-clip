import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Import custom modules
from geoclip.train.dataloader import GeoDataLoader, img_train_transform, img_val_transform
from geoclip.train.train import train 
from geoclip.train.eval import eval_images
from geoclip.model.GeoCLIP import GeoCLIP


def main():
    # Hyperparameters
    batch_size = 4
    num_workers = 8
    num_epochs = 300
    learning_rate = 3e-5
    wd=1e-6
    gamma=0.87
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = pd.read_csv("/home/rvl/Documents/rvl/pohsun/datasets/taipei.csv")
    ds = ds.sample(frac=1).reset_index(drop=True)
    frac = int(len(ds)*0.8)
    print(f"frac={frac}")
    train_ds = ds[:frac] 
    val_ds = ds[frac:]

    # Dataset and DataLoader
    train_dataset = GeoDataLoader(train_ds, "", transform=img_train_transform())
    val_dataset = GeoDataLoader(val_ds, "", transform=img_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize GeoCLIP model
    model = GeoCLIP(from_pretrained=False, queue_size=4096)
    model.to(device)

    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    distance_errors = []
    angle_errors = []

    # Training loop
    for epoch in range(num_epochs):
        train(
            train_loader,
            model,
            optimizer,
            epoch,
            batch_size,
            device,
            scheduler,
            criterion,
        )

        # Evaluation
        distance_error, heading_angle_error = eval_images(val_loader, model, device)
        print(f"distance error: {distance_error} px, heading angle error: {heading_angle_error}")
        # print(f"Epoch {epoch+1}/{num_epochs} completed. Best distance error: {coordinate_error:.4f}. Best heading angle error: {heading_angle_error:.4f}, accuracy: {best_accuracy:.4f}")

    print("Training completed.")

    print(distance_errors.shape)
    print(angle_errors.shape)

    fig, (ax1, ax2) = plt.subplot(2, 1, figsize=(10, 4))
    # Plot the first subplot
    ax1.plot(len(distance_errors), distance_errors)
    ax1.set_title('Distance error')
    ax1.set_xlabel('distance(pixel)')
    ax1.set_ylabel('errors')

    # Plot the second subplot
    ax2.plot(360, angle_errors)
    ax2.set_title('Heading angle errors')
    ax2.set_xlabel('degree')
    ax2.set_ylabel('errors')

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
