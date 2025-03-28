import numpy as np
from icecream import ic
import seaborn as sns
from itertools import product


def old_pick_neighbor(spins: np.ndarray, i: int, j: int, N: int, ksize: int):
    neighbors_bound = ksize // 2
    use_range = np.arange(-neighbors_bound, neighbors_bound + 1)
    neighbors = [(i + r1, j + r2) for (r1, r2) in product(use_range, use_range)]
    neighbors_mat = np.array(neighbors).reshape((ksize, ksize, -1))
    vectorized_mod = np.vectorize(lambda x: x % N)
    neighbors_mat = vectorized_mod(neighbors_mat)
    neighbors = [(i + r1, j + r2) for (r1, r2) in product(use_range, use_range)]
    neighbors_mat = np.array(neighbors).reshape((ksize, ksize, -1))
    vectorized_mod = np.vectorize(lambda x: x % N)
    neighbors_mat = vectorized_mod(neighbors_mat)
    picked = np.empty((ksize, ksize))
    for _i, _j in product(range(ksize), range(ksize)):
        picked[_i, _j] = spins[*neighbors_mat[_i, _j]]
    return picked


def generate_neighbor_indices(
    center: np.ndarray, distances: np.ndarray, dim_bounds: np.ndarray
) -> np.ndarray:
    _c = np.tile(center.reshape(-1, *([1] * len(center))), reps=2 * distances + 1)
    ic(_c)
    _base = np.indices(2 * distances + 1, dtype=int)
    ic(_base)
    _neighbor = np.tile(
        distances.reshape(-1, *(len(center) * [1])), reps=2 * distances + 1
    )
    ic(_neighbor)
    _translator = _base + _c - _neighbor
    ic(_base.shape)
    ic(np.sum(_translator))
    ic(np.sum(_translator[0]))
    ic(np.sum(_translator[1]))

    for dim in range(len(center)):
        _translator[dim, :] = _translator[dim, :] % dim_bounds[dim]
    ic(_translator.shape)
    return _translator


def pick_neighbor(
    data: np.ndarray, center: np.ndarray, distances: np.ndarray
) -> np.ndarray:
    picker = generate_neighbor_indices(center, distances, data.shape)
    ic(picker[0])
    ic(picker[1])
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

    fig, axes = plt.subplots(ncols=3, figsize=(18, 20))
    sns.heatmap(full_pic, ax=axes[0], annot=True)
    center = np.array([2, 2])
    distances = np.array([1, 1])
    picked = pick_neighbor(full_pic, center, distances)
    ic(picked.shape)
    sns.heatmap(picked, ax=axes[1], annot=True)
    o_picked = old_pick_neighbor(full_pic, center[0], center[1], 8, 3)
    sns.heatmap(o_picked, ax=axes[2], annot=True)
    plt.show()


if __name__ == "__main__":
    test()
