import torch

def normalization(x: torch.Tensor):
    return (x - x.mean()) / x.std(0)

def min_max_normalization(x: torch.Tensor):

    return (x - x.min()) / (x.max() - x.min())

def negative_normalization(x: torch.Tensor):
    normalized = (x - x.min()) / (x.max() - x.min())
    return 2 * normalized - 1
