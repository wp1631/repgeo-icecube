from scipy.stats import norm, vonmises
import numpy as np
from scipy.special import i0, i1
import matplotlib.pyplot as plt


def vonmises_to_gaussian_stdev(kappa):
    """
    Calculate the equivalence standard deviation of gaussian distribution to the vonmises distribution
    """
    return (np.pi / 2) * np.sqrt(1 - i1(kappa) / i0(kappa))


def main():
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.flatten()
    kappa_range = np.logspace(-2, 2, 30)
    ax[0].plot(kappa_range, vonmises_to_gaussian_stdev(kappa_range))
    ax[0].set_xlabel("kappa")
    ax[0].set_ylabel("stdev")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_title("stdev/channel_distance vs. kappa")

    stim_range = np.linspace(-1, 1, 100)
    norm_stds = []
    von_stds = []
    for kappa in kappa_range:
        probe_random_sample_num = 1000000
        norm_dat = norm(loc=0, scale=vonmises_to_gaussian_stdev(kappa)).rvs(
            size=probe_random_sample_num
        )
        von_dat = vonmises(loc=0, kappa=kappa).rvs(size=probe_random_sample_num)
        norm_stds.append(np.std(norm_dat))
        von_stds.append(np.std(von_dat))
    ax[1].plot(kappa_range, norm_stds, c="r", label="norm_std")
    ax[1].plot(kappa_range, von_stds, c="b", label="kappa_std")
    ax[1].plot(kappa_range, np.array(norm_stds) / np.array(von_stds), label="ratio")
    ax[1].set_xscale("log")
    ax[1].legend()
    ax[1].set_title("Standard deviation ratio of gaussian and vonmises")
    plt.show()


if __name__ == "__main__":
    main()
