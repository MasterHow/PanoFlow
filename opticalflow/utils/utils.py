import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class InputPadder:
    """Pads images such that dimensions are divisible by 8 , from RAFT."""

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [
                pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2,
                pad_ht - pad_ht // 2
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def fill_order_keys(key, fill_value='_model.', fill_position=7):
    """fill order_dict keys in checkpoint, by Hao."""
    return key[0:fill_position] + fill_value + key[fill_position:]


def fix_order_keys(key, delete_value=6):
    """fix order_dict keys in checkpoint, by Hao."""
    return key[0:delete_value] + key[13:]


def fix_read_order_keys(key, start_value=7):
    """fix reading restored ckpt order_dict keys, by Hao."""
    return key[start_value:]


# CARLA semantic labels
camvid_colors = OrderedDict([
    ("Unlabeled", np.array([0, 0, 0], dtype=np.uint8)),
    ("Building", np.array([70, 70, 70], dtype=np.uint8)),
    ("Fence", np.array([100, 40, 40], dtype=np.uint8)),
    ("Other", np.array([55, 90, 80], dtype=np.uint8)),
    ("Pedestrian", np.array([220, 20, 60], dtype=np.uint8)),
    ("Pole", np.array([153, 153, 153], dtype=np.uint8)),
    ("RoadLine", np.array([157, 234, 50], dtype=np.uint8)),
    ("Road", np.array([128, 64, 128], dtype=np.uint8)),
    ("SideWalk", np.array([244, 35, 232], dtype=np.uint8)),
    ("Vegetation", np.array([107, 142, 35], dtype=np.uint8)),
    ("Vehicles", np.array([0, 0, 142], dtype=np.uint8)),
    ("Wall", np.array([102, 102, 156], dtype=np.uint8)),
    ("TrafficSign", np.array([220, 220, 0], dtype=np.uint8)),
    ("Sky", np.array([70, 130, 180], dtype=np.uint8)),
    ("Ground", np.array([81, 0, 81], dtype=np.uint8)),
    ("Bridge", np.array([150, 100, 100], dtype=np.uint8)),
    ("RailTrack", np.array([230, 150, 140], dtype=np.uint8)),
    ("GroundRail", np.array([180, 165, 180], dtype=np.uint8)),
    ("TrafficLight", np.array([250, 170, 30], dtype=np.uint8)),
    ("Static", np.array([110, 190, 160], dtype=np.uint8)),
    ("Dynamic", np.array([170, 120, 50], dtype=np.uint8)),
    ("Water", np.array([45, 60, 150], dtype=np.uint8)),
    ("Terrain", np.array([145, 170, 100], dtype=np.uint8)),
])


def convert_label_to_grayscale(im):
    out = (np.ones(im.shape[:2]) * 255).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_pxls = np.where((im == np.asarray(rgb)).sum(-1) == 3)
        out[match_pxls] = gray_val
    assert (out !=
            255).all(), "rounding errors or missing classes in camvid_colors"
    return out.astype(np.uint8)


def convert_label_to_rgb(im):
    out = np.zeros((im.shape[0], im.shape[1], 3)).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_x, match_y = np.where(im == gray_val)
        out[match_x, match_y] = rgb
    return out.astype(np.uint8)
