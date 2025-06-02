import numpy as np
from scipy.stats import special_ortho_group
from log import get_logger

logger = get_logger()


def expand_coord(data: np.ndarray, target_dim: int):
    if data.shape[1] > target_dim:
        logger.warning(
            "Target dimension < original dimension -> Using original dimension"
        )
        return data
    _expanded = np.zeros((data.shape[0], target_dim))
    _expanded[:, : data.shape[1]] = data
    return _expanded


def rotate(data: np.ndarray):
    _dim = data.shape[1]
    _rotation_matrix = special_ortho_group.rvs(_dim)
    return np.matmul(data, _rotation_matrix)
