import numpy as np
from icecream import ic
import seaborn as sns


def generate_neighbor_indices(
    center: np.ndarray, distances: np.ndarray, dim_bounds: np.ndarray
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
    full_pic = np.fromfunction(
        lambda x, y: np.cos(0.02 * (x + y) + 0.0002 * x**2 - 0.0004 * y**2),
        shape=(8, 8),
    )
    ic(full_pic.shape)
    # full_pic = np.asarray(Image.open(r"/home/waragonp/Documents/GitHub/repgeo-icecube/src/utils/test_img.jpeg").convert("L")).astype(np.float128).copy()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(ncols=2, figsize=(18, 12))
    sns.heatmap(full_pic, ax=axes[0], annot=True)

    center = np.array([7, 0])
    distances = np.array([2, 2])
    picked = pick_neighbor(full_pic, center, distances)
    ic(picked.shape)
    sns.heatmap(picked, ax=axes[1], annot=True)
    plt.show()


if __name__ == "__main__":
    test()
