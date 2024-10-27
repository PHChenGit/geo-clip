from os import walk
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from .loss import orientation_loss


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
    orientation_errors = []

    with torch.no_grad():
        gps_gallery = model.gps_gallery.to(device)
        for images, true_coords, orientation in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            true_coords = true_coords.to(device)
            orientation = orientation.to(device)

            with torch.autocast(device_type="cuda"):
                pred_logits, logits_orientation = model(images, gps_gallery)
            probs = pred_logits.softmax(dim=-1)
            pred_indices = torch.argmax(probs, dim=-1)

            pred_coords = gps_gallery[pred_indices]

            distances = torch.norm(pred_coords - true_coords, dim=1)
            total_distance += distances.sum().item()

            total_samples += images.size(0)
            
            _, angle_error = orientation_loss(logits_orientation, orientation)
            orientation_errors.append(angle_error)

    # 计算平均距离误差
    mean_distance_error = total_distance / total_samples

    orientation_errors = torch.cat(orientation_errors, dim=0)
    mae = torch.mean(orientation_errors).item()
    print(f"Average distance error (pixel)：{mean_distance_error:.4f}")
    print(f"Orientation MAE (pixel)：{mae:.4f}")

    return mean_distance_error, mae
