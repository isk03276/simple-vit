from typing import Union

import numpy as np
import torch
import patchify


def pad_image(image: Union[np.ndarray, torch.Tensor]):
    pass

def slice_image_to_patches(images: torch.Tensor, patch_size: int)-> torch.Tensor:
    """
    Split images into patches. 
    Assume that images have shape of (N * C * H * W).
    """
    assert isinstance(images, torch.Tensor)
    assert len(images.shape) == 4
    
    images_shape = images.shape
    n_batch, n_channel = images_shape[:2]
    patches = images.unfold(1, n_channel, n_channel).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).squeeze()
    patches = patches.reshape(n_batch, -1, n_channel, patch_size, patch_size)
    return patches
