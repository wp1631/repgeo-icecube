import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
import pathlib
from utils.generators.neurons import intialize_spins
from utils.solvers.metropolis_hastling import metropolis_hastings_mexican_hat
from utils.icecream import ic, IC_ENABLED
from utils.savers.images import save_img
from utils.savers.parameters import save_params

N = 500  # Grid size
J = 1.0  # Interaction strength
T = 0.005  # Temperature
SAVE_EVERY = 100_000
random_seed = np.random.randint(1000000)
np.random.seed(random_seed)
num_steps = int(10_000_000)  # Number of Metropolis stepus
ksize = 35
sigma_pos = 7
sigma_neg = 10
orientation_dev_range = np.pi / 10

PARAMETER_DICT = {
    "T": T,
    "N": N,
    "J": J,
    "ksize": ksize,
    "seed": random_seed,
    "sigma_pos": sigma_pos,
    "sigma_neg": sigma_neg,
    "numstep": num_steps,
    "metropolis_orientation_dev": orientation_dev_range,
}

spins = intialize_spins(size=(N, N), seed=random_seed, vmin=0, vmax=np.pi)
timestamp = datetime.now()
hashed_ts = hashlib.md5(timestamp.isoformat().encode()).hexdigest()
folder_name = hashed_ts
base_folder = pathlib.Path().cwd().joinpath("data").joinpath(folder_name)
base_folder.mkdir(parents=True)
save_params(PARAMETER_DICT, base_folder)
data_folder = base_folder.joinpath("dat")
data_folder.mkdir(parents=True, exist_ok=True)
img_folder = base_folder.joinpath("img")
img_folder.mkdir(parents=True, exist_ok=True)


def save_data(num):
    if (num + 1) % 10000000 == 0:
        savefile_dat = data_folder.joinpath(
            f"N{N}_J{J}_T{T}_ITER{num}_KSIZE{ksize}_PSIGMA{sigma_pos}_NSIGMA{sigma_neg}_TS{datetime.now()}.npy"
        )
        savefile_img = img_folder.joinpath(
            f"N{N}_J{J}_T{T}_ITER{num}_KSIZE{ksize}_PSIGMA{sigma_pos}_NSIGMA{sigma_neg}_TS{datetime.now()}.jpg"
        )
        np.save(savefile_dat, spins)
        save_img(savefile_img, spins)


def main():

    metropolis_hastings_mexican_hat(
        spins,
        J,
        T,
        ksize,
        sigma_pos,
        sigma_neg,
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
                f"N{N}_J{J}_T{T}_ITER{num_steps}_KSIZE{ksize}_PSIGMA{sigma_pos}_NSIGMA{sigma_neg}_TS{datetime.now()}.jpg"
            )
        )
    except:
        print("Save error")
    plt.show()


if __name__ == "__main__":
    if not IC_ENABLED:
        ic.disable()
    main()
