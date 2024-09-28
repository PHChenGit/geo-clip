import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic as GD

def rvl_loss(targets, preds):
    def cal_coordinate():
        t = [[target[0] for target in targets], [target[1] for target in targets]]
        p = [[pred[0] for pred in preds],  [pred[1] for pred in preds]]
        rmse = np.sqrt(np.square(np.subtract(t, p)).mean()) 
        return rmse

    def cal_angle():
        t_angle = np.array([target[2] for target in targets])
        p_angle = np.array([pred[2] for pred in preds])
        angle = (t_angle - p_angle + 180) % 360 - 180
        return angle

    coordinate_error = cal_coordinate()
    heading_angle_error = cal_angle()
    return alpha * coordinate_error + beta * heading_angle_error

def eval_images(val_dataloader, model, device="cpu"):
    model.eval()
    model.to(device)

    gps_gallery = model.gps_gallery
    gps_gallery = gps_gallery.to(device)

    gt_x = torch.tensor([])
    gt_y = torch.tensor([])
    gt_angles = torch.tensor([])

    pred_x = torch.tensor([])
    pred_y = torch.tensor([])
    pred_angles = torch.tensor([])

    with torch.no_grad():
        for imgs_A, labels in tqdm(val_dataloader, desc="Evaluating"):
            gt_x = torch.cat([gt_x, labels[0]])
            gt_y = torch.cat([gt_y, labels[1]])
            gt_angles = torch.cat([gt_angles, labels[2]])

            labels = torch.stack(labels, dim=1)
            labels = labels.cpu().numpy()
            imgs_A = imgs_A.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image, theta_pred, t_x_pred, t_y_pred= model(imgs_A, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)
            
            # Predict gps location with the highest probability (index)
            # outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            # preds = gps_gallery[outs].detach().cpu().numpy()
            outs = torch.argmax(probs, dim=-1)
            preds = gps_gallery[outs].detach().cpu()
            # print("preds")
            # print(preds[:, 0])

            pred_x = torch.cat([pred_x, preds[:, 0]])
            # print("pred_x")
            # print(pred_x)
            pred_y = torch.cat([pred_y, preds[:, 0]])
            pred_angles = torch.cat([pred_angles, preds[:, 0]])
            # pred_x.cat(preds[:, 0])
            # pred_y.append(preds[:, 1])
            # pred_angles.append(preds[:, 2])

    distance = torch.sqrt(torch.square(torch.subtract(gt_x, pred_x)) + torch.square(torch.subtract(gt_y, pred_y))).mean()
    angles = (torch.subtract(gt_angles, pred_angles) + 180) % 360 - 180
    avg_angles = angles.mean()
    return distance, avg_angles 

