import torch
from torch import nn
import torch.nn.functional as F
# from torch.cuda.amp import autocast as autocast, GradScaler
import numpy as np
from tqdm import tqdm

def train(train_dataloader, model, optimizer, epoch, total_epoch, batch_size, device, scaler, scheduler=None, criterion=nn.CrossEntropyLoss()):
    # print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)

    epoch_contrastive_loss = 0.0
    epoch_orientation_loss = 0.0
    epoch_total_loss = 0.0
    num_batches = 0

    for i ,(imgs, gps, orientation) in bar:
        imgs = imgs.to(device)
        gps = gps.to(device)
        orientation = orientation.to(device)
        gps_queue = model.get_gps_queue()

        optimizer.zero_grad()

        # Append GPS Queue & Queue Update
        gps_all = torch.cat([gps, gps_queue], dim=0).to(device)
        model.dequeue_and_enqueue(gps)

        # Prepare ground truth orientation
        # Convert angles from degrees to radians
        orientation_rad = orientation * (np.pi / 180.0)
        # Compute sin θ and cos θ
        sin_theta = torch.sin(orientation_rad)
        cos_theta = torch.cos(orientation_rad)
        orientation_gt = torch.stack((sin_theta, cos_theta), dim=1).to(device)

        # Forward pass
        with torch.autocast(device_type="cuda"): 
            logits_img_gps, orientation_pred= model(imgs, gps_all)

            # Compute the loss
            img_gps_loss = criterion(logits_img_gps, targets_img_gps)
            # orientation_loss = torch.mean(((orientation_gt - orientation_pred + 180) % 360 - 180).square(), dtype=torch.float32).sqrt()
            # orientation_loss = torch.min(torch.abs(orientation_gt - orientation_pred).mean(), 360 - torch.abs(orientation_gt - orientation_pred).mean())
            mse_loss = nn.MSELoss()
            orientation_loss = mse_loss(orientation_pred, orientation_gt)
            orientation_loss = torch.sqrt(orientation_loss)

            alpha = 0.1 # weighting factor for orientation loss
            loss = img_gps_loss + alpha * orientation_loss 

        # Backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        # Accumulate losses
        epoch_contrastive_loss += img_gps_loss.item()
        epoch_orientation_loss += orientation_loss.item()
        epoch_total_loss += loss.item()
        num_batches += 1

        bar.set_description("Epoch {}/{} loss: {:.5f}".format(epoch, total_epoch, loss.item()))

    if scheduler is not None:
        scheduler.step()

    return epoch_contrastive_loss, epoch_orientation_loss, epoch_total_loss, num_batches
