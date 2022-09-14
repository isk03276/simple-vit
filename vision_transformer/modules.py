from ast import Mult
from unittest.mock import patch
import torch
import torch.nn as nn

from vision_transformer.layers import MultiHeadAttention, MLPBlock
from utils.image import slice_image_to_patches


class PatchEmbedder(nn.Module):
    def __init__(self, image_size: int, n_channel: int, n_patch: int, n_dim: int):
        super().__init__()
        self.n_patch = n_patch
        self.linear_projection = nn.Linear(n_channel * n_patch**2, n_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, n_dim))
        self.position_embedding = nn.Parameter(torch.randn((image_size//n_patch) ** 2 + 1, n_dim))
    
    def forward(self, x):
        # (B, C, H, W) -> (B, N, C  * P * P)
        patches = slice_image_to_patches(images=x, patch_size=self.n_patch, flatten=True)
        batch_size = patches.shape[0]
        embs = self.linear_projection(patches)
        # (1, 1, dim) -> (B, 1, dim )
        class_token = self.class_token.repeat(batch_size, 1, 1)
        embs = torch.cat((embs, class_token), dim = 1)
        embs += self.position_embedding
        return embs


class EncoderBlock(nn.Module):
    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_dim)
        self.multi_head_attention = MultiHeadAttention(n_dim=n_dim, n_heads=n_heads)
        self.mlp_block = MLPBlock(n_dim=n_dim)
        
    def forward(self, x):
        x_backup = x
        x = self.layer_norm(x)
        x = self.multi_head_attention(x)
        x = x_backup = x + x_backup
        x = self.layer_norm(x)
        x = self.mlp_block(x)
        x += x_backup
        return x
        
        
    