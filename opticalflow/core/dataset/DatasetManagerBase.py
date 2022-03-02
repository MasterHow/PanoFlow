# The interface class for all datasets
import abc

import numpy as np
from torch.utils.data import DataLoader


class DatasetManagerBase(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        pass

    @abc.abstractclassmethod
    def load_images(root_dir: str, **kwargs):
        """Load Images from directory and return a numpy array."""
        pass

    @abc.abstractclassmethod
    def _preprocess_image(cls, image: np.ndarray) -> np.ndarray:
        """Preprocess single image."""
        pass

    @abc.abstractclassmethod
    def create_dataloader(cls, images: np.ndarray) -> DataLoader:
        pass

    @abc.abstractclassmethod
    def _postprocess_image(cls, image: np.ndarray) -> np.ndarray:
        """Postprocess single image."""
        pass

    @abc.abstractclassmethod
    def _preprocess_flow(cls, image: np.ndarray) -> np.ndarray:
        """Preprocess single flow image."""
        pass

    @abc.abstractclassmethod
    def preprocess_data(cls, data):
        """Preprocess I/O images."""
        pass

    @abc.abstractclassmethod
    def postprocess_data(cls, data):
        """Postprocess output images."""
        pass
