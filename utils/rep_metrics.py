import numpy as np
from scipy.spatial.distance import pdist, squareform
import numpy.typing as npt
from typing import Optional


def center_gram(G):
    """Center the Gram matrix."""
    n = G.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ G @ H


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two observation matrices.

    Parameters:
    - X: ndarray of shape (n_observations, n_features1)
    - Y: ndarray of shape (n_observations, n_features2)

    Returns:
    - cka: float, similarity between X and Y
    """
    # Compute Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # Centered Gram matrices
    K_centered = center_gram(K)
    L_centered = center_gram(L)

    # Compute Hilbert-Schmidt Independence Criterion (HSIC)
    hsic = np.sum(K_centered * L_centered)

    # Normalization terms
    norm_x = np.linalg.norm(K_centered, "fro")
    norm_y = np.linalg.norm(L_centered, "fro")

    return hsic / (
        norm_x * norm_y + 1e-10
    )  # Add small constant to prevent divide by zero


def global_distance_variance(X: npt.NDArray[np.floating], Y: npt.NDArray[np.floating]):
    assert len(X) == len(Y)
    X_pdist = pdist(X)
    X_pdist_normed = X_pdist / np.max(X_pdist)
    Y_pdist = pdist(Y)
    Y_pdist_normed = Y_pdist / np.max(Y_pdist)
    diff = X_pdist_normed - Y_pdist_normed
    diff_sq = np.power(diff, 2)
    sum_sq = np.sqrt(np.sum(diff_sq))
    return sum_sq / np.sqrt(X.shape[1])


def global_neigbor_dice(
    X: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
    *,
    sort_index: Optional[npt.NDArray] = None,
    dist_ratio_threshold: float = 20,
):
    assert len(X) == len(Y)
    l = len(X)
    if sort_index:
        use_X, use_Y = X[sort_index], Y[sort_index]
    else:
        use_X, use_Y = X, Y
    X_pdist = pdist(use_X)
    X_threshold = np.percentile(
        X_pdist[X_pdist > np.min(X_pdist)], dist_ratio_threshold
    )
    Y_pdist = pdist(use_Y)
    Y_threshold = np.percentile(
        Y_pdist[Y_pdist > np.min(Y_pdist)], dist_ratio_threshold
    )
    X_conn = squareform(X_pdist < X_threshold) + np.identity(n=l)
    Y_conn = squareform(Y_pdist < Y_threshold) + np.identity(n=l)
    dices = []
    for i in range(l):
        _X, _Y = X_conn[i], Y_conn[i]
        its = np.sum(np.logical_and(X_conn[i], Y_conn[i]))
        tot = np.sum(_X) + np.sum(_Y)
        dices.append(2 * its / tot)
    return dices
