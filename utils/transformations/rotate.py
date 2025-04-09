from scipy.stats import special_ortho_group
import numpy as np


def rotate(data: np.ndarray):
    _dim = data.shape[1]
    _rotation_matrix = special_ortho_group.rvs(_dim)
    return np.matmul(data, _rotation_matrix)
