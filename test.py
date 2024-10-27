import os
import cv2
import random

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your model and dataset classes
from geoclip.model.GeoCLIP import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_test_transform

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def visualize(output):
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    image_path = "../../datasets/pohsun/taipei_satellite_imgs/Taipie_202408.jpg"

    if not os.path.exists(os.path.join(image_path)):
        print(f"Image file not exists: {image_path}")
        exit(-1)

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
        center_1 = (
            int(true_locations[idx][0].cpu().item()),
            int(true_locations[idx][1].cpu().item()),
        )
        center_2 = (
            int(predicted_locations[idx][0].cpu().item()),
            int(predicted_locations[idx][1].cpu().item()),
        )
        cv2.line(img, center_1, center_2, COLOR_BLUE, thickness=thickness)

    cv2.imwrite(os.path.join(output, "test_result.png"), img)


def distance_acc(lo_1, lo_2):
    error = torch.sub(lo_1, lo_2)
    error = torch.square(error)
    error = torch.add(error[0], error[1])
    error = torch.sqrt(error)
    return error

def orientation_loss(pred, target):
    N = torch.divide(pred, 360)
    pred = torch.where(pred >= 360, pred - (N * 360), pred)
    pred = torch.where(pred < 0, pred + (torch.abs(N) * 360), pred)
    
    error1 = torch.abs(target - pred)
    error2 = torch.sub(360, error1)
    error = torch.min(error1, error2)
    return error


device = "cuda" if torch.cuda.is_available() else "cpu"

model = GeoCLIP(from_pretrained=True)
model.load_state_dict(
    torch.load(
        os.path.join(
            "./output/",
            "20241024_regression_sin_cos",
            "geoclip_gpt_img_size_326.pth",
        )
    )
)

model.to(device)
model.eval()  # Set model to evaluation mode

# Load test data
test_dataset = GeoDataLoader(
    dataset_file="../datasets/with_angle_2/test/taipei.csv",
    dataset_folder="../datasets/with_angle_2/test",
    transform=img_test_transform(),
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

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
        locations = locations.cpu()
        orientations = orientations.cpu()

        with torch.autocast(device_type="cuda"): 
            pred_location, pred_prob, pred_orientation = model.predict(images, top_k=1)
        predicted_locations.extend(pred_location)
        true_locations.extend(locations)

        predicted_orientations.extend(pred_orientation)
        true_orientations.extend(orientations)

        num_samples += 1
        # total_location_error += (locations - pred_location).cpu().item()
        # total_orientation_error += (orientations - pred_orientation).cpu().item()


location_errors = []
for idx in range(len(true_locations)):
    location_errors.append(distance_acc(true_locations[idx], predicted_locations[idx]))

orientation_errors = []
for idx in range(len(true_orientations)):
    error = orientation_loss(predicted_orientations[idx], true_orientations[idx])
    # error = (true_orientations[idx] - predicted_orientations[idx] + 180) % 360 - 180
    orientation_errors.append(error.cpu().item())

output_dir = "./output/20241024_regression_sin_cos/test"
orientation_df = pd.DataFrame(orientation_errors)
orientation_df.to_csv(os.path.join(output_dir, "orientation_errors.csv"), header=None, index=False)
# Calculate average errors
avg_location_error = np.mean(location_errors)
orientation_mae = np.mean(orientation_errors)

print(f"Average Location Error: {avg_location_error:.4f}")
print(f"Orientation MAE (degree): {orientation_mae:.4f} degrees")

samples = range(1, num_samples + 1)

plt.figure(figsize=(10, 5))
plt.plot(samples, location_errors, label="Predict Distance Error")
plt.xlabel("Samples")
plt.ylabel("Error (pixels)")
plt.title("Testing Distance Metrics over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "test_distance_image_size_326.png"))
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(samples, orientation_errors, label="Orientation Error")
plt.xlabel("Epoch")
plt.ylabel("Error (degree)")
plt.title("Evaluation Orientation over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "test_orientation_image_size_326.png"))
plt.show()

visualize(output_dir)
