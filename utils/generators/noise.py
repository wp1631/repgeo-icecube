import numpy as np


def create_block_noise(
    block_size=5, total_size=50, observation: int = 3, minor_amp: float = 0.1
):
    _noise = np.empty((observation, total_size))
    _n = total_size // block_size
    for i in range(_n):
        _noise[:, block_size * i : block_size * (i + 1)] = np.random.normal(
            0, 1, (observation, 1)
        )
        _noise[
            :, block_size * i : block_size * (i + 1)
        ] += minor_amp * np.random.normal(0, 1, (observation, block_size))
    l = _noise[:, block_size * (_n) :].shape[1]
    if l > 0:
        _noise[:, block_size * (_n) :] = np.random.normal((observation, 1))
        _noise[:, block_size * (_n) :] += minor_amp * np.random.normal((observation, l))
    return _noise
