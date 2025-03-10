from typing import Literal
import numpy as np
from numpy.linalg import cholesky

BLOCK_SIZE = (30, 8)
REPEAT_NUM = 5


def _generate_col_block(
    size: tuple[int, int],
    mean: np.ndarray | float | int = 0.0,
    covariance: np.ndarray | None = None,
):
    if covariance is None:
        covariance = np.identity(n=size[1])
    l = cholesky(covariance)
    z = np.random.normal(loc=0, scale=1, size=size)
    if isinstance(mean, np.ndarray):
        assert mean.shape == (0, size[1])
        return mean + z @ l
    elif isinstance(mean, (float, int)):
        mean = np.full(shape=(1, size[1]), fill_value=mean)
        return mean + z @ l
    raise NotImplementedError()


def _generate_row_block(
    size: tuple[int, int],
    mean: np.ndarray | float | int = 0.0,
    covariance: np.ndarray | None = None,
):
    if covariance is None:
        covariance = np.identity(n=size[0])
    print("begin")
    l = cholesky(covariance)
    print("cov")
    z = np.random.normal(loc=0, scale=1, size=size)
    if isinstance(mean, np.ndarray):
        assert mean.shape == (0, size[0])
        return mean + (z.T @ l.T).T
    elif isinstance(mean, (float, int)):
        mean = np.full(shape=(size[0], 1), fill_value=mean)
        return mean + (z.T @ l.T).T
    raise NotImplementedError()


def generate_block(
    size: tuple[int, int],
    mean: np.ndarray | float | int = 0,
    covariance: np.ndarray | None = None,
    mode: Literal["col", "row"] = "col",
):
    handle = _generate_row_block
    if mode == "col":
        handle = _generate_col_block
    return handle(size, mean, covariance)


if __name__ == "__main__":
    pass
