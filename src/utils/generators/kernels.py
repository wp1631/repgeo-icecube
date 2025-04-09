from functools import lru_cache
import numpy as np
import cv2


@lru_cache
def get_gaussian_kernel(ksize: int, sigma: float = 1, dims=2) -> np.ndarray:
    gaussian_filter_1d = cv2.getGaussianKernel(ksize, sigma=sigma)
    if dims == 1:
        return gaussian_filter_1d
    if dims == 2:
        gaussian_filter_2d = gaussian_filter_1d @ gaussian_filter_1d.T
        gaussian_filter_2d[int(ksize / 2), int(ksize / 2)] = 0
        return gaussian_filter_2d
    else:
        raise NotImplemented
