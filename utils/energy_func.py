import numpy as np


def cosine_similarity_energy(
    J: float, center: float, spins: np.ndarray, interaction_kernel: np.ndarray
) -> float:
    return -J * np.sum(np.cos((center - spins) * 2) * interaction_kernel)
