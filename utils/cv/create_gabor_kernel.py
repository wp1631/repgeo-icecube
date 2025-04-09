import cv2
import numpy as np
from icecream import ic

ic.configureOutput(includeContext=True)
from typing import Literal, TypeAlias

Numeric: TypeAlias = int | float


def getGaborKernel(
    size: tuple[int, int],
    sigma: Numeric,
    ori: Numeric,
    wavelength: Numeric,
    spatial_ratio: Numeric = 1,
    phase: Numeric = 0,
    angle_unit: Literal["rad", "deg"] = "deg",
) -> np.ndarray:
    """
    This function create a Gabor Kernel as np.ndarray from the gabor function specification!

    """
    for idx, num in enumerate(size):
        if not (num % 2):
            ic(f"size {size[idx]} at position {idx+1} should be changed to odd number")

    (use_ori, use_phase) = (
        (np.deg2rad(ori), np.deg2rad(phase)) if angle_unit == "deg" else (ori, phase)
    )
    kernel = cv2.getGaborKernel(
        size, sigma, use_ori, wavelength, spatial_ratio, use_phase, ktype=cv2.CV_32F
    )
    return kernel


if __name__ == "__main__":
    pass
