import numpy as np
from icecream import ic

ic.configureOutput(includeContext=True)
from typing import TypeAlias

Numeric: TypeAlias = int | float


def crop_image(
    image: np.ndarray, loc: tuple[int, int], size: tuple[int, int]
) -> np.ndarray:
    """
    crop image from the image np.ndarray (x,y,channel)
    """

    return image[loc[0] : loc[0] + size[0], loc[1] : loc[1] + size[1]]


if __name__ == "__main__":
    pass
