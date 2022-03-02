import numpy as np

from opticalflow.core.data_aug import RadialDistortion


def __build_distortion(method: str, **kwargs):
    if method == 'radial':
        if not kwargs:
            kwargs = {'ks': [0, 1e-5, 0, 1e-14, 0, 1e-15]}
        distortion = RadialDistortion(**kwargs)
    else:
        raise NotImplementedError(f'Unknown method: {method}')
    return distortion


def distort_img(tensor: np.ndarray,
                method: str = 'radial',
                resolution=None,
                nearest=None,
                inverse=False,
                **kwargs) -> np.ndarray:
    distortion = __build_distortion(method, **kwargs)
    return distortion.distort_img(tensor, output_resolution=resolution,  nearest_inter=nearest, inverse=inverse)


def distort_flow(tensor: np.ndarray,
                 method: str = 'radial',
                 resolution=None,
                 nearest=None,
                 **kwargs) -> np.ndarray:
    distortion = __build_distortion(method, **kwargs)
    return distortion.distort_flow(tensor, output_resolution=resolution,  nearest_inter=nearest)
