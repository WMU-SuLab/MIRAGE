import numpy as np
import torch

def tensor2numpy(tensor: torch.Tensor, mean: list[float, ...] = [0.485, 0.456, 0.406],
                 std: list[float, ...] = [0.229, 0.224, 0.225], ):
    image = tensor.to('cpu').clone().detach().numpy().squeeze()
    image = image.transpose((1, 2, 0))
    image = image * np.array(std) + np.array(mean)
    image = image.clip(0, 1)
    return image
