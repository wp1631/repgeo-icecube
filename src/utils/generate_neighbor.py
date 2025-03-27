from cv2 import repeat
import numpy as np
from typing import Sequence
from icecream import ic


def generate_neighbor_indices(
    center: np.ndarray, distances: np.array, dim_bounds: np.ndarray
) -> np.ndarray:
    _c = np.tile(center.reshape(-1, *([1] * len(center))), reps=2 * distances + 1)
    ic(_c.shape)
    _base = np.indices(2 * distances + 1, dtype=int)
    _neighbor = np.tile(
        distances.reshape(-1, *(len(center) * [1])), reps=2 * distances + 1
    )
    _translator = _base - _neighbor
    ic(_base.shape)
    ic(np.sum(_translator))
    for dim in range(len(center)):
        _translator[dim, :] = _translator[dim, :] % dim_bounds[dim]
    ic(_translator.shape)
    return _translator


def pick_neighbor(
    data: np.ndarray, center: np.ndarray, distances: np.ndarray
) -> np.ndarray:
    picker = generate_neighbor_indices(center, distances, data.shape)
    picked = data[picker[0], [picker[1]]].squeeze()
    ic(picked.shape)
    return picked


def test():
    full_pic = np.tile(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ],
        reps=[20, 1],
    ).astype(np.float128)
    full_pic += 2 * np.random.random(size=full_pic.shape)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(full_pic, vmax=15)
    center = np.array([7, 0])
    full_pic[center] = 100
    distances = np.array([4, 8])
    picked = pick_neighbor(full_pic, center, distances)
    ic(picked.shape)
    axes[1].imshow(picked, vmax=15)
    plt.show()


if __name__ == "__main__":
    test()
