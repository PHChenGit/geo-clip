import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np



def train(train_dataloader, model, optimizer, epoch, batch_size, device, scheduler=None, criterion=nn.CrossEntropyLoss()):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)
    lambda_reg=0.1

    for _ ,(imgs_A, coordinates) in bar:
        imgs_A = imgs_A.to(device)
        c_x_gt = coordinates[0]
        c_y_gt = coordinates[1]
        theta_gt = coordinates[2]

        c_x_gt = c_x_gt.to(device)
        c_y_gt = c_y_gt.to(device)
        theta_gt = theta_gt.to(device)
        coordinates = torch.stack(coordinates, dim=1) # (x, y, theta)

        coordinates = coordinates.to(device)
        gps_queue = model.get_gps_queue()

        optimizer.zero_grad()

        # Append GPS Queue & Queue Update
        gps_all = torch.cat([coordinates, gps_queue], dim=0)
        model.dequeue_and_enqueue(coordinates)

        # Forward pass
        logits_img_gps, theta_pred, t_x_pred, t_y_pred = model(imgs_A, gps_all) # similarity matrix

        # Compute the loss
        img_gps_loss = criterion(logits_img_gps, targets_img_gps)

        # 计算损失
        total_loss, contrastive_loss, regression_loss = model.compute_loss(
            logits_img_gps, theta_pred, t_x_pred, t_y_pred,
            targets_img_gps, theta_gt, c_x_gt, c_y_gt, lambda_reg=lambda_reg
        )
        loss = total_loss

        # Backpropagate
        loss.backward()
        optimizer.step()

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    if scheduler is not None:
        scheduler.step()
