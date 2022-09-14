import numpy as np
import torch


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()
