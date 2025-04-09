from utils.generate_neighbor import pick_neighbor
from utils.generators.kernels import get_gaussian_kernel
import numpy as np
from utils.icecream import ic
from utils.energy_func import cosine_similarity_energy as energy_func
from typing import cast, Callable


def _metropolis_hastings_step(
    spins: np.ndarray,
    J: float,
    T: float,
    interaction_weight_kernel_sigma: float,
    kernel_size: int,
    spin_deviation_sigma: float,
) -> None:
    random_index = np.array([np.random.randint(dim_size) for dim_size in spins.shape])
    ic(random_index)
    assert kernel_size % 2 != 0
    n_bound = int(kernel_size // 2)
    interaction_kernel = get_gaussian_kernel(
        kernel_size, interaction_weight_kernel_sigma
    )
    ic(interaction_kernel.shape)
    picked = pick_neighbor(
        spins,
        center=random_index,
        distances=np.full(len(spins.shape), fill_value=n_bound),
    )
    centre = cast(float, spins[*random_index])
    new_centre = (
        cast(float, np.random.uniform(-spin_deviation_sigma, spin_deviation_sigma))
        + centre
    )
    energy_old = energy_func(J, centre, picked, interaction_kernel)
    energy_new = energy_func(J, new_centre, picked, interaction_kernel)
    delta_E = energy_new - energy_old
    if delta_E < 0 or np.exp(-delta_E / T) > np.random.rand():
        spins[*random_index] = new_centre % np.pi  # Accept new state


def metropolis_hastings(
    spins: np.ndarray,
    interaction_J: float = 1.0,
    temperature: float = 0.005,
    interaction_kernel_size: int = 17,
    interaction_sigma: float = 5,
    spin_deviation_sigma: float = np.pi / 10,
    steps: int = 100000,
    save_callback: Callable | None = None,
) -> np.ndarray:
    ic(spins.shape)
    assert len(spins.shape) == 2
    i: int = 0
    while i < steps:
        _metropolis_hastings_step(
            spins,
            interaction_J,
            temperature,
            interaction_sigma,
            interaction_kernel_size,
            spin_deviation_sigma,
        )
        if save_callback:
            save_callback(i)
        i += 1
    return spins
