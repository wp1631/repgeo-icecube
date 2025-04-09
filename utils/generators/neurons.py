from typing import Sequence
import numpy as np


def intialize_spins(
    size: Sequence[int],
    seed: int,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    np.random.seed(seed)
    return np.random.uniform(vmin, vmax, size)
