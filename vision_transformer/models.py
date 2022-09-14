import torch
import torch.nn as nn

from vision_transformer.modules import PatchEmbedder, EncoderBlock
from vision_transformer.layers import Classifier
from utils.image import slice_image_to_patches


class ViT(nn.Module):
    def __init__(self, image_size: int, n_channel: int, n_patch: int, n_dim: int, n_encoder_blocks: int, n_heads: int, n_classes: int):
        super().__init__()
        self.patch_embedder = PatchEmbedder(image_size=image_size, n_channel=n_channel, n_patch=n_patch, n_dim=n_dim)
        self.encoder = EncoderBlock(n_dim=n_dim, n_heads = n_heads)
        self.classifier = Classifier(n_dim=n_dim, n_classes=n_classes)
        
    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0])
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
