import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import hashlib
import pathlib
import json
from icecream import ic
from typing import Sequence
from utils.generate_neighbor import pick_neighbor
from typing import Callable
from functools import lru_cache

IC_ENABLED = False


def intialize_spins(
    size: Sequence[int],
    seed: int,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    np.random.seed(seed)
    return np.random.uniform(vmin, vmax, size)


@lru_cache
def get_gaussian_kernel(ksize: int, sigma: float = 1, dims=2) -> np.ndarray:
    gaussian_filter_1d = cv2.getGaussianKernel(ksize, sigma=sigma)
    if dims == 1:
        return gaussian_filter_1d
    if dims == 2:
        gaussian_filter_2d = gaussian_filter_1d @ gaussian_filter_1d.T
        gaussian_filter_2d[int(ksize / 2), int(ksize / 2)] = 0
        return gaussian_filter_2d
    else:
        raise NotImplemented


def energy_func(
    J: float, center: float, spins: np.ndarray, interaction_kernel: np.ndarray
) -> float:
    return -J * np.sum(np.cos((center - spins) * 2) * interaction_kernel)


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
    centre = spins[*random_index]
    ic(centre.shape)
    new_centre = np.random.uniform(-spin_deviation_sigma, spin_deviation_sigma) + centre
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


def save_img(savefile: pathlib.Path, data: np.ndarray):
    use_ori = np.abs(data * 180 / np.pi)
    _ = plt.imshow(use_ori, cmap="hsv", vmin=0, vmax=180)
    _ = plt.colorbar()
    _ = plt.savefig(savefile)
    plt.close()
    return


def main():
    # Parameters
    N = 500  # Grid size
    J = 1.0  # Interaction strength
    T = 0.005  # Temperature
    random_seed = np.random.randint(1000000)
    np.random.seed(random_seed)
    num_steps = int(1e6)  # Number of Metropolis stepus
    ksize = 17
    sigma = 5
    spins = np.random.uniform(0, np.pi, (N, N))
    orientation_dev_range = np.pi / 10
    timestamp = datetime.now()
    hashed_ts = hashlib.md5(timestamp.isoformat().encode()).hexdigest()
    folder_name = hashed_ts
    base_folder = pathlib.Path().cwd().joinpath("data").joinpath(folder_name)
    base_folder.mkdir(parents=True)

    with open(base_folder.joinpath("params.json"), "w") as file:
        params = {
            "T": T,
            "N": N,
            "J": J,
            "ksize": ksize,
            "seed": random_seed,
            "sigma": sigma,
            "numstep": num_steps,
            "metropolis_orientation_dev": orientation_dev_range,
        }
        json.dump(params, file, indent=4)

    data_folder = base_folder.joinpath("dat")
    data_folder.mkdir(parents=True, exist_ok=True)
    img_folder = base_folder.joinpath("img")
    img_folder.mkdir(parents=True, exist_ok=True)

    def save_data(num):
        if (num + 1) % 10000 == 0:
            savefile_dat = data_folder.joinpath(
                f"N{N}_J{J}_T{T}_ITER{num}_KSIZE{ksize}_SIGMA{sigma}_TS{datetime.now()}.npy"
            )
            savefile_img = img_folder.joinpath(
                f"N{N}_J{J}_T{T}_ITER{num}_KSIZE{ksize}_SIGMA{sigma}_TS{datetime.now()}.jpg"
            )
            np.save(savefile_dat, spins)
            save_img(savefile_img, spins)

    metropolis_hastings(
        spins,
        J,
        T,
        ksize,
        sigma,
        orientation_dev_range,
        steps=num_steps,
        save_callback=save_data,
    )

    use_ori = np.abs(spins * 180 / np.pi)
    plt.imshow(use_ori, cmap="hsv", vmin=0, vmax=180)
    plt.colorbar()
    try:
        plt.savefig(
            img_folder.joinpath(
                f"N{N}_J{J}_T{T}_ITER{num_steps}_KSIZE{ksize}_SIGMA{sigma}_TS{datetime.now()}.jpg"
            )
        )
    except:
        print("Save error")
    plt.show()


if __name__ == "__main__":
    if not IC_ENABLED:
        ic.disable()
    main()
