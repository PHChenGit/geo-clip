import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=True, queue_size=4096):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder(from_pretrained=from_pretrained)

        self.gps_gallery = load_gps_data(
            os.path.join(
                "/home/rvl/Documents/rvl/pohsun/datasets/taipei_coordinate_8946.csv"
            )
        )
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = "cuda"
        # 添加回归头
        self.regression_head = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 输出3个值：theta^*, t_x^*, t_y^*
        )

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(
            torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth")
        )
        self.location_encoder.load_state_dict(
            torch.load(f"{self.weights_folder}/location_encoder_weights.pth")
        )
        self.logit_scale = nn.Parameter(
            torch.load(f"{self.weights_folder}/logit_scale_weights.pth")
        )

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(3, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)

        assert (
            self.queue_size % gps_batch_size == 0
        ), f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr : gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()

    def forward(self, image, location):
        """GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        image_features_normalized = F.normalize(image_features, dim=1)
        location_features_normalized = F.normalize(location_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features_normalized @ location_features_normalized.t())

        # 回归预测
        regression_output = self.regression_head(image_features)
        theta_pred = regression_output[:, 0]
        t_x_pred = regression_output[:, 1]
        t_y_pred = regression_output[:, 2]

        return logits_per_image, theta_pred, t_x_pred, t_y_pred

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob

    def angular_loss(self, predicted, target):
        # 将角度转换为弧度
        predicted_rad = torch.deg2rad(predicted)
        target_rad = torch.deg2rad(target)

        # 计算角度误差
        loss = torch.abs(torch.atan2(torch.sin(predicted_rad - target_rad), torch.cos(predicted_rad - target_rad)))

        # 将误差转换回角度
        loss = torch.rad2deg(loss)
        return loss.mean()

    def compute_loss(self, logits_per_image, theta_pred, t_x_pred, t_y_pred, labels, theta_gt, t_x_gt, t_y_gt, lambda_reg=0.1):
        # 对比学习的损失
        contrastive_loss = F.cross_entropy(logits_per_image, labels)

        # 回归损失
        theta_loss = self.angular_loss(theta_pred, theta_gt)
        t_x_loss = F.l1_loss(t_x_pred, t_x_gt)
        t_y_loss = F.l1_loss(t_y_pred, t_y_gt)
        regression_loss = theta_loss + t_x_loss + t_y_loss

        # 总损失
        total_loss = contrastive_loss + lambda_reg * regression_loss

        return total_loss, contrastive_loss, regression_loss

