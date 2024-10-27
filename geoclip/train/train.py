import torch
from torch import nn
import torch.nn.functional as F
# from torch.cuda.amp import autocast as autocast, GradScaler
import numpy as np
from tqdm import tqdm

from .loss import orientation_loss


def train(train_dataloader, model, optimizer, epoch, total_epochs, batch_size, device, scaler, scheduler=None, criterion=nn.CrossEntropyLoss()):
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

        # Forward pass
        with torch.autocast(device_type="cuda"): 
            logits_img_gps, logits_orientation = model(imgs, gps_all)

            # Compute the loss
            img_gps_loss = criterion(logits_img_gps, targets_img_gps)
            angle_mae, angle_error = orientation_loss(logits_orientation, orientation)
            alpha = 0.1 # weighting factor for orientation loss
            loss = img_gps_loss + alpha * angle_mae.item()

        # Backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        # Accumulate losses
        epoch_contrastive_loss += img_gps_loss.item()
        epoch_orientation_loss += angle_mae.item()
        epoch_total_loss += loss.item()
        num_batches += 1

        bar.set_description(f"Epoch {epoch}/{total_epochs} loss: {loss.item():.5f}")

    if scheduler is not None:
        scheduler.step()

    return epoch_contrastive_loss, epoch_orientation_loss, epoch_total_loss, num_batches
