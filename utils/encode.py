import torch


def classifications_convert_to_one_hot(classifications, num_classes):

    one_hot = torch.zeros(classifications.size(0), num_classes)
    one_hot.scatter_(1, classifications.view(-1, 1), 1)
    return one_hot
