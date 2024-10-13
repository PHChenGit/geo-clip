import os
import cv2
import random

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your model and dataset classes
from geoclip.model.GeoCLIP import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_test_transform


def visualize():
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    image_path = "/home/rvl/Documents/rvl/datasets/pohsun/taipei_satellite_imgs/Taipie_202408.jpg" 
    sat_img = cv2.imread(image_path)
    img = sat_img.copy()
    rad = 10
    thickness = 10

    for location in true_locations:
        center = (int(location[0].cpu().item()), int(location[1].cpu().item()))
        cv2.circle(img, center, rad, COLOR_GREEN, thickness)

    for location in predicted_locations:
        center = (int(location[0].cpu().item()), int(location[1].cpu().item()))
        cv2.circle(img, center, rad, COLOR_RED, thickness)

    for idx in range(len(true_locations)):
        center_1 = (int(true_locations[idx][0].cpu().item()), int(true_locations[idx][1].cpu().item()))
        center_2 = (int(predicted_locations[idx][0].cpu().item()), int(predicted_locations[idx][1].cpu().item()))
        cv2.line(img, center_1, center_2, COLOR_BLUE, thickness=thickness)

    cv2.imwrite("./output/test_result.png", img)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GeoCLIP(from_pretrained=True)
model.load_state_dict(torch.load(os.path.join("/home/rvl/Documents/rvl/pohsun/geo-clip/output/", "geoclip_gpt_img_size_350.pth")))

model.to(device)
model.eval()  # Set model to evaluation mode

# Load test data
test_dataset = GeoDataLoader(
    dataset_file='/home/rvl/Documents/rvl/pohsun/datasets/with_angle/test/taipei_100.csv',
    dataset_folder='/home/rvl/Documents/rvl/pohsun/datasets/with_angle/test',
    transform=img_test_transform()
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

# Evaluation metrics
total_location_error = 0.0
total_orientation_error = 0.0
num_samples = 0

# Lists to store results for plotting or further analysis
predicted_locations = []
true_locations = []
predicted_orientations = []
true_orientations = []

bar = tqdm(enumerate(test_loader), total=len(test_loader))
with torch.no_grad():
    for _, (images, locations, orientations) in bar:
        # images = images.to(device)
        locations = locations.to(device)
        orientations = orientations.to(device)

        pred_location, pred_prob, pred_orientation = model.predict(images, top_k=1)
        predicted_locations.extend(pred_location)
        true_locations.extend(locations)

        predicted_orientations.extend(pred_orientation)
        true_orientations.extend(orientations)

        num_samples += 1


# Calculate average errors
avg_location_error = total_location_error / num_samples
avg_orientation_error = total_orientation_error / num_samples

print(f"Average Location Error: {avg_location_error:.4f}")
print(f"Average Orientation Error: {avg_orientation_error:.4f} degrees")

visualize()
