import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
# from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from geoclip.model.GeoCLIP import GeoCLIP
from geoclip.train.train import train
from geoclip.train.eval import eval_images, evaluate_rmse
from geoclip.train.dataloader import GeoDataLoader, img_train_transform, img_val_transform

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    parser = argparse.ArgumentParser(description='训练 GeoCLIP 模型（自定义数据集）')
    parser.add_argument('--dataset_csv', type=str, help='数据集的 CSV 文件路径')
    parser.add_argument('--dataset_folder', type=str, help='包含图像的数据集文件夹路径')
    parser.add_argument('--batch_size', type=int, default=4, help='训练的批次大小')
    parser.add_argument('--num_epochs', type=int, default=500, help='训练的轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--queue_size', type=int, default=4096, help='队列的大小')
    parser.add_argument('--output_dir', type=str, default='output', help='保存检查点和日志的目录')
    parser.add_argument('--device', type=str, default='cuda', help='用于训练的设备（cuda 或 cpu）')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载的工作线程数')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集所占比例（0-1之间）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = GeoCLIP(from_pretrained=False, queue_size=args.queue_size)
    model.to(device)

    train_ds = GeoDataLoader(
        dataset_file='/home/rvl/Documents/rvl/pohsun/datasets/with_angle/train/taipei.csv',
        dataset_folder='/home/rvl/Documents/rvl/pohsun/datasets/with_angle/train',
        transform=img_train_transform()
    )
    val_ds = GeoDataLoader(
        dataset_file='/home/rvl/Documents/rvl/pohsun/datasets/with_angle/val/taipei.csv',
        dataset_folder='/home/rvl/Documents/rvl/pohsun/datasets/with_angle/val',
        transform=img_val_transform()
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = None
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    distance_errors = []
    rmses = []
    orientation_rmses = []

    contrastive_losses = []
    orientation_losses = []
    total_losses = []

    gps_gallery_path = os.path.join("/home/rvl/Documents/rvl/pohsun/datasets/with_angle/", "gps_gallery.csv")
    if not os.path.exists(gps_gallery_path):
        all_gps = []
        for _, gps, _ in tqdm(train_dataloader, desc="gps gallery"):
            all_gps.append(gps)

        all_gps = torch.cat(all_gps, dim=0)
        df = pd.DataFrame({
            "LAT": all_gps[:, 0].tolist(),
            "LON": all_gps[:, 1].tolist(),
        })
        df.to_csv(gps_gallery_path, index=False)

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, args.num_epochs + 1):
        epoch_contrastive_loss, epoch_orientation_loss, epoch_total_loss, num_batches = train(
            train_dataloader,
            model,
            optimizer,
            epoch,
            args.batch_size,
            device,
            scaler,
            scheduler,
            criterion
        )

        # Calculate average losses for the epoch
        avg_contrastive_loss = epoch_contrastive_loss / num_batches
        avg_orientation_loss = epoch_orientation_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches

        contrastive_losses.append(avg_contrastive_loss)
        orientation_losses.append(avg_orientation_loss)
        total_losses.append(avg_total_loss)

        # eval_images(val_dataloader, model, device)
        mean_distance_error, orientaion_rmse = evaluate_rmse(val_dataloader, model, device)

        distance_errors.append(mean_distance_error)
        # rmses.append(rmse)
        orientation_rmses.append(orientaion_rmse)

        # self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))
        # self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth"))
        # self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))
        checkpoint_path = os.path.join(args.output_dir, f'geoclip_gpt_img_size_350.pth')
        torch.save(model.state_dict(), checkpoint_path)

        if scheduler is not None:
            scheduler.step()

    epochs = range(1, args.num_epochs+1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, label='Total Loss', color='blue', marker='o')
    plt.plot(epochs, contrastive_losses, label='Contrastive Loss', color='red', marker='o')
    plt.plot(epochs, orientation_losses, label='Orientation Loss', color='green', marker='o')

    plt.title('Loss Trends Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, distance_errors, label='Average Distance Error')
    # plt.plot(epochs, rmses, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Error (pixels)')
    plt.title('Evaluation Metrics over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'evaluation_metrics_image_size_350.png'))
    plt.show()

    np.save('./output/orientation_rmses.npy', orientation_rmses)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, orientation_rmses, label='Orientation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Error (degree)')
    plt.title('Evaluation Orientation over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'evaluation_orientation_image_size_350.png'))
    plt.show()

if __name__ == '__main__':
    main()

