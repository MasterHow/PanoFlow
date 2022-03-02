import cv2
import cvbase as cvb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .DatasetManagerBase import DatasetManagerBase


class KITTIDataset(Dataset):

    def __init__(self, images: torch.tensor) -> None:
        super().__init__()
        self._images = images
        self._len = len(images)

    def __len__(self):
        return self._len

    def __getitem__(self, index) -> torch.tensor:
        return self._images[index]


class KITTIDemoManager(DatasetManagerBase):

    IMG1_SUFFIX = '_10.png'
    IMG2_SUFFIX = '_11.png'
    FLOW_SUFFIX_KITTI = '_flow_10.png'
    FLOW_SUFFIX = '_10.flo'
    BATCH_SIZE = 1

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def load_images(cls, img_prefix: str, args):
        """Load Images from directory and return a numpy array."""
        # load imgs
        img1 = cv2.imread(img_prefix + cls.IMG1_SUFFIX).astype(np.float32)
        img2 = cv2.imread(img_prefix + cls.IMG2_SUFFIX).astype(np.float32)

        # load flow
        if args.dataset == 'KITTI':
            flow = cv2.imread(img_prefix + cls.FLOW_SUFFIX_KITTI,
                              cv2.IMREAD_ANYDEPTH
                              | cv2.IMREAD_COLOR).astype(np.float32)
        else:
            flow = cvb.read_flow(img_prefix + cls.FLOW_SUFFIX)

        return (img1, img2), flow

    @classmethod
    def _preprocess_image(cls, image: np.ndarray, args) -> np.ndarray:
        """Preprocess single image."""
        if args.model == 'RAFT':
            image = image.transpose(2, 0, 1)
        return image

    @classmethod
    def _preprocess_flow(cls, image: np.ndarray, args) -> np.ndarray:
        """Preprocess single flow image."""
        if args.dataset == 'KITTI':
            image = image[:, :, ::-1].astype(np.float32)
            flow, _ = image[:, :, :2], image[:, :, 2]
            flow = (flow - 2**15) / 64.0
            # flow is a (u, v) 2-channel image, u, v \in [-512, 512]
        else:
            flow = image
        return flow

    @classmethod
    def create_dataloader(cls, images, args) -> DataLoader:
        (x1, x2), y = images
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)
        y = torch.from_numpy(y)

        x1 = x1.to(args.DEVICE)
        x2 = x2.to(args.DEVICE)
        y = y.to(args.DEVICE)

        images = (x1, x2), y

        dataset = KITTIDataset([images])
        dataloader = DataLoader(dataset, batch_size=cls.BATCH_SIZE)
        return dataloader

    @classmethod
    def preprocess_data(cls, data, args):
        """Preprocess I/O images."""
        (img1, img2), gt = data
        img1 = cls._preprocess_image(img1, args)
        img2 = cls._preprocess_image(img2, args)
        gt = cls._preprocess_flow(gt, args)
        return (img1, img2), gt

    @classmethod
    def postprocess_data(cls, data):
        """Postprocess output images."""
        data = data.to('cpu').numpy()
        return cls._postprocess_image(data)
