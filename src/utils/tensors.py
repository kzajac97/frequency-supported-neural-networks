import numpy as np
import torch


def torch_to_flat_array(tensor: torch.Tensor) -> np.array:
    """Converts torch Tensor (could be on GPU) to flat numpy array"""
    return tensor.detach().cpu().numpy().flatten()
