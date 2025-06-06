import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from itertools import product
from datetime import datetime
import hashlib
import pathlib
import json

# Parameters
N = 500  # Grid size
J = 1.0  # Interaction strength
T = 0.005  # Temperature
random_seed = np.random.randint(1000000)
np.random.seed(random_seed)
num_steps = int(1e6)  # Number of Metropolis stepus
ksize = 17
sigma = 5
neighbors_bound = int(ksize // 2)
use_range = np.arange(-neighbors_bound, neighbors_bound + 1)
orientation_dev_range = np.pi / 10
i = 0
j = 0
neighbors = [(i + r1, j + r2) for (r1, r2) in product(use_range, use_range)]
neighbors_mat = np.array(neighbors).reshape((ksize, ksize, -1))
vectorized_mod = np.vectorize(lambda x: x % N)
neighbors_mat = vectorized_mod(neighbors_mat)
gaussian_filter_1d = cv2.getGaussianKernel(ksize, sigma=sigma)
gaussian_filter_2d = gaussian_filter_1d @ gaussian_filter_1d.T
gaussian_filter_2d[int(ksize / 2), int(ksize / 2)] = 0
picked = np.empty((ksize, ksize))
energy_old = 0
energy_new = 0
spins = np.random.uniform(0, np.pi, (N, N))


def save_img(savefile: pathlib.Path, data: np.ndarray):
    use_ori = np.abs(data * 180 / np.pi)
    _ = plt.imshow(use_ori, cmap="hsv", vmin=0, vmax=180)
    _ = plt.colorbar()
    _ = plt.savefig(savefile)
    plt.close()
    return


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

for num in range(num_steps):
    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
    neighbors = [(i + r1, j + r2) for (r1, r2) in product(use_range, use_range)]
    neighbors_mat = np.array(neighbors).reshape((ksize, ksize, -1))
    vectorized_mod = np.vectorize(lambda x: x % N)
    neighbors_mat = vectorized_mod(neighbors_mat)
    picked = np.empty((ksize, ksize))
    for _i, _j in product(range(ksize), range(ksize)):
        picked[_i, _j] = spins[*neighbors_mat[_i, _j]]
    centre = picked[int(ksize / 2), int(ksize / 2)]
    new_centre = (
        np.random.uniform(-orientation_dev_range, orientation_dev_range) + centre
    )
    # energy_old = -J * np.sum(np.cos(centre - picked) * gaussian_filter_2d)
    # energy_new = -J * np.sum(np.cos(new_centre - picked) * gaussian_filter_2d)
    energy_old = -J * np.sum(np.cos((centre - picked) * 2) * gaussian_filter_2d)
    energy_new = -J * np.sum(np.cos((new_centre - picked) * 2) * gaussian_filter_2d)
    delta_E = energy_new - energy_old
    if (num + 1) % 10000 == 0:
        savefile_dat = data_folder.joinpath(
            f"N{N}_J{J}_T{T}_ITER{num}_KSIZE{ksize}_SIGMA{sigma}_TS{datetime.now()}.npy"
        )
        savefile_img = img_folder.joinpath(
            f"N{N}_J{J}_T{T}_ITER{num}_KSIZE{ksize}_SIGMA{sigma}_TS{datetime.now()}.jpg"
        )
        np.save(savefile_dat, spins)
        save_img(savefile_img, spins)
    if delta_E < 0 or np.exp(-delta_E / T) > np.random.rand():
        spins[i, j] = new_centre % np.pi  # Accept new state
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
