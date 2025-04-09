import numpy as np


def expand_coord(data: np.ndarray, target_dim: int):
    assert data.shape[1] <= target_dim
    _expanded = np.zeros((data.shape[0], target_dim))
    _expanded[:, : data.shape[1]] = data
    return _expanded


if __name__ == "__main__":
    ...
