from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence

import cv2
import numpy as np


class Distortion(metaclass=ABCMeta):
    """Abstract class for distorction."""

    def __init__(self):
        pass

    @abstractmethod
    def distort_pos_func(self, pos: np.ndarray, resolution: tuple,
                         inverse: bool) -> np.ndarray:
        """pos (np.ndarray): The position of each point in grid. resolution
        (tuple): The input resolution(height, width) in tuple. inverse (bool):
        If inverse is `True`, the function will map the distorted position to
        calibrated position. This parameter is useful in image calibration.

        Returns:
            np.ndarray: The distorted position of each point in grid.
        """
        pass

    def distort_img_pos(self,
                        height: int,
                        width: int,
                        pos_arr: np.ndarray = None,
                        inverse: bool = False) -> np.ndarray:
        """Given a array of position, return their distortion position.

        By default, the pos_arr comprises the coordinate of all pixels of an
        image.
        """
        if pos_arr is None:
            ij_grid = np.indices((height, width), dtype=np.float32)
            pos_arr = np.transpose(ij_grid, (1, 2, 0))

        new_pos = self.distort_pos_func(pos_arr, (height, width), inverse)
        new_pos = new_pos.astype(np.float32)
        return new_pos

    def distort_img(self,
                    img: np.ndarray,
                    inverse: bool = False,
                    output_resolution: Optional[tuple] = None,
                    nearest_inter: bool = None) -> np.ndarray:
        assert img.ndim >= 2
        height, width = img.shape[0], img.shape[1]

        distort_pos = self.distort_img_pos(height, width, inverse=inverse)

        distort_pos = np.flip(distort_pos, 2)
        if nearest_inter:
            res = cv2.remap(img, distort_pos, None, cv2.INTER_NEAREST)
        else:
            res = cv2.remap(img, distort_pos, None, cv2.INTER_LINEAR)
        res = res.astype(img.dtype)
        if output_resolution:
            output_resolution = np.flip(output_resolution, 0)
            if nearest_inter:
                res = cv2.resize(
                    res, output_resolution, interpolation=cv2.INTER_NEAREST)
            else:
                res = cv2.resize(
                    res, output_resolution, interpolation=cv2.INTER_LINEAR)
        return res

    def distort_flow(self,
                     flow: np.ndarray,
                     inverse: bool = False,
                     output_resolution: Optional[tuple] = None,
                     nearest_inter: bool = None) -> np.ndarray:
        height, width = flow.shape[0], flow.shape[1]
        i_indices = np.arange(height, dtype=np.float32)
        j_indices = np.arange(width, dtype=np.float32)
        ji_grid = np.meshgrid(j_indices, i_indices)
        ji_grid = np.transpose(ji_grid, (1, 2, 0))
        ji_grid -= flow
        dst_pos = np.flip(ji_grid, 2)

        distort_pos = self.distort_img_pos(height, width, inverse=inverse)

        distort_origin_pos = self.distort_img_pos(
            height, width, inverse=not inverse)
        distort_dst_pos = self.distort_img_pos(
            height, width, dst_pos, inverse=not inverse)
        flow_d = (distort_origin_pos - distort_dst_pos).astype(np.float32)
        distort_pos = np.flip(distort_pos, 2)
        if nearest_inter:
            res = cv2.remap(flow_d, distort_pos, None, cv2.INTER_NEAREST)
        else:
            res = cv2.remap(flow_d, distort_pos, None, cv2.INTER_LINEAR)
        res = res.astype(np.float32)
        res = np.flip(res, 2)

        if output_resolution:
            ratio = np.array(output_resolution) / np.array(flow.shape[0:2])
            output_resolution = np.flip(output_resolution, 0)
            if nearest_inter:
                res = cv2.resize(
                    res, output_resolution, interpolation=cv2.INTER_NEAREST)
            else:
                res = cv2.resize(
                    res, output_resolution, interpolation=cv2.INTER_LINEAR)
            ratio = np.flip(ratio, 0)
            np.multiply(res, ratio, res)
        return res


class RadialDistortion(Distortion):

    def __init__(self, ks: Sequence[float]):
        """
        A sixtic polynominal L(r) = 1 + k_1r + k_2r^2 + k_4r^4 + k_6r^6

        Args:
            ks(Sequence[float]): The parameters of k_1, k_2, ...
        """
        super().__init__()
        self.ks = ks

    def distort_func(self, r: np.ndarray):
        res = 1
        r_pow = 1
        for k in self.ks:
            r_pow *= r
            res += k * r_pow

        return res

    def distort_pos_func(self, pos: np.ndarray, resolution: tuple,
                         inverse: bool) -> np.ndarray:
        """pos (np.ndarray): The position of each point in grid. resolution
        (tuple): The input resolution(height, width) in tuple. inverse (bool):
        If inverse is `True`, the function will map the distorted position to
        calibrated position. This parameter is useful in image calibration.

        Returns:
            np.ndarray: The distorted position of each point in grid.
        """
        height, width = resolution

        center = np.array([height / 2, width / 2], dtype=pos.dtype)
        dis_ij = pos - center
        r = np.linalg.norm(dis_ij, axis=2)
        Lr: np.ndarray = self.distort_func(r)
        Lr = Lr.reshape(*Lr.shape, 1)
        if inverse:
            np.divide(dis_ij, Lr, dis_ij)
        else:
            np.multiply(dis_ij, Lr, dis_ij)
        np.add(dis_ij, center, dis_ij)

        return dis_ij
