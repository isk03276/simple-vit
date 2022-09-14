import numpy as np
import torch


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def get_device(device_name: str) -> torch.device:
    try:
        device = torch.device(device_name)
    except RuntimeError as e:
        print("[Device name error] Use cpu device!")
        device = torch.device("cpu")
    return device
