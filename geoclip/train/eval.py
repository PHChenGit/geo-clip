from os import walk
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm


def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    avg_distance_error = 0

    for i in range(total):
        pred_coord = gps_gallery[preds[i]].detach().cpu().numpy()
        true_coord = targets[i]
        distance = np.linalg.norm(pred_coord - true_coord)
        avg_distance_error += distance
        if distance <= dis:
            correct += 1

    avg_distance_error /= total
    accuracy = correct / total
    return accuracy, avg_distance_error

def eval_images(val_dataloader, model, device="cpu"):
    model.eval()

    all_gps = []
    for _, gps in val_dataloader:
        all_gps.append(gps)
    all_gps = torch.cat(all_gps, dim=0).to(device)
    model.set_gps_gallery(all_gps)

    preds = []
    targets = []

    gps_gallery = model.gps_gallery

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)
            
            # Predict gps location with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            
            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1] # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        accuracy_results[f'acc_{dis}_km'] = acc

    return accuracy_results

def evaluate_rmse(dataloader, model, device="cpu"):
    model.eval()
    total_distance = 0.0
    total_samples = 0
    squared_errors = []
    orientation_errors = []

    with torch.no_grad():
        gps_gallery = model.gps_gallery.to(device)
        for images, true_coords, orientation in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            true_coords = true_coords.to(device)
            orientation = orientation.to(device)

            pred_logits, orientation_pred = model(images, gps_gallery)
            probs = pred_logits.softmax(dim=-1)
            pred_indices = torch.argmax(probs, dim=-1)

            pred_coords = gps_gallery[pred_indices]

            distances = torch.norm(pred_coords - true_coords, dim=1)
            total_distance += distances.sum().item()

            # squared_error = (pred_coords - true_coords) ** 2
            # squared_errors.append(squared_error.detach().cpu())

            total_samples += images.size(0)

            # Prepare ground truth orientation
            # Convert angles from degrees to radians
            orientation_rad = orientation * (np.pi / 180.0)
            # Compute sin θ and cos θ
            sin_theta = torch.sin(orientation_rad)
            cos_theta = torch.cos(orientation_rad)
            orientation_gt = torch.stack((sin_theta, cos_theta), dim=1).to(device)

            # Orientation Loss (MSE Loss between predicted and ground truth sin and cos values)
            orientation_error = ((orientation_gt - orientation_pred + 180) % 360 - 180).square()
            orientation_errors.append(orientation_error)

    # 计算平均距离误差
    mean_distance_error = total_distance / total_samples

    # # 计算 RMSE
    # squared_errors = torch.cat(squared_errors, dim=0)
    # mse = squared_errors.mean().item()
    # rmse = np.sqrt(mse)

    print(f"Average distance error (pixel)：{mean_distance_error:.4f}")
    # print(f"RMSE (pixel)：{rmse:.4f}")

    orientation_errors = torch.cat(orientation_errors, dim=0)
    mse = orientation_errors.mean()
    heading_angle_error = torch.sqrt(mse).item()

    return mean_distance_error, heading_angle_error
