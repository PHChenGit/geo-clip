import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoProcessor

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class OrientationEncoder(nn.Module):

    def __init__(self):
        super(OrientationEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x)
        x = self.rotation_head(x)

        return x 