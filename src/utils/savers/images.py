import pathlib
import matplotlib.pyplot as plt
import numpy as np


def save_img(savefile: pathlib.Path, data: np.ndarray):
    use_ori = np.abs(data * 180 / np.pi)
    _ = plt.imshow(use_ori, cmap="hsv", vmin=0, vmax=180)
    _ = plt.colorbar()
    _ = plt.savefig(savefile)
    plt.close()
    return
