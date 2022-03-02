import torch

from opticalflow.core.dataset import KITTIDemoManager


def postprocess_data(input_imgs, result: torch.tensor):
    result = KITTIDemoManager.postprocess_data(result)
    return result
