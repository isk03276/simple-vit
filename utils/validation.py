import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.torch import tensor_to_array


def visualize_one_tensor_image(image: torch.Tensor):
    """
    Visualize images. Assume that images have shape of (C, H, W)
    """
    assert isinstance(image, torch.Tensor)
    assert len(image.shape) == 3

    np_image = tensor_to_array(image.permute(1, 2, 0))
    plt.imshow(np_image)
    plt.show()


def visualize_multiple_tensor_images(images: torch.Tensor):
    """
    Visualize images. Assume that images have shape of (C, H, W)
    """
    assert isinstance(images, torch.Tensor)
    assert len(images.shape) == 4

    np_images = tensor_to_array(images.permute(0, 2, 3, 1))
    fig, axs = plt.subplots(ncols=len(np_images), squeeze=False)
    for i, image in enumerate(np_images):
        axs[0, i].imshow(np.asarray(image))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
