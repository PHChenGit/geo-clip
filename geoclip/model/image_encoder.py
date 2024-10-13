import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoProcessor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

        # Orientation prediction head
        self.orientation_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output sin θ and cos θ
        )

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x)
        x = self.mlp(x)

        # Normalize image features
        image_features = F.normalize(x, dim=1)

        # Orientation prediction
        orientation_pred = self.orientation_head(x)
        # Normalize to ensure outputs lie on the unit circle
        orientation_pred = F.normalize(orientation_pred, dim=1)
        return x, orientation_pred
