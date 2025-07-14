from typing import Optional
import numpy as np
from scipy.stats import vonmises
from scipy.special import i0
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from utils.generators.classes_1D import Stimulus1D
import matplotlib as mpl
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def get_max_vonmises(kappa: float):
    return np.exp(kappa) / (2 * np.pi * i0(kappa))


# plot neural tuning profile
def plot_neural_orientation_tuning_profile(
    stimulus: Stimulus1D,
    neuron_tuning_loc: np.ndarray,
    neuron_tuning_kappa: np.ndarray,
    neuron_tuning_amp: np.ndarray,
    /,
    dpi: int = 200,
    plot_every: int = 300,
    alpha: float = 0.2,
    cmap: str = "viridis",
    ax: Optional[Axes] = None,
):
    _ax = ax
    if not ax:
        fig, _ax = plt.subplots(dpi=dpi)
    probe_stim = np.sort(stimulus.orientation)
    num_lines = 1 + len(neuron_tuning_loc) // plot_every
    _center = num_lines // 2
    _cmap = mpl.colormaps[cmap]
    _colors = _cmap(np.linspace(0, 1, num_lines))
    for tuning_loc, tuning_kappa, tuning_amp, col in zip(
        neuron_tuning_loc[::plot_every],
        neuron_tuning_kappa[::plot_every],
        neuron_tuning_amp[::plot_every],
        _colors,
    ):
        _ax.plot(
            probe_stim,
            tuning_amp * vonmises.pdf(probe_stim, loc=tuning_loc, kappa=tuning_kappa),
            alpha=alpha,
            c=col,
        )
    _ax.plot(
        probe_stim,
        neuron_tuning_amp[_center * plot_every]
        * vonmises.pdf(
            probe_stim,
            loc=neuron_tuning_loc[_center * plot_every],
            kappa=neuron_tuning_kappa[_center * plot_every],
        ),
        alpha=1,
        c=_colors[_center],
    )
    _ax.set_title("Neural Tuning Function")
    _ax.set_xlabel("Orientation")
    _ax.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\pi /2$", "$-\pi /4$", "0", "$\pi/4$", "$\pi/2$"],
    )
    _ax.set_ylim(
        0, np.max(get_max_vonmises(np.max(neuron_tuning_kappa[::plot_every])) * 1.1)
    )
    if not ax:
        plt.show()


def plot_orientation_activation(
    neural_responses: np.ndarray,
    sort_index: np.ndarray,
    /,
    cmap: str = "binary",
    xlabel: str = "neuron_id",
    ylabel: str = "stimulus",
    title: str = "Neural Activation vs Stimulus",
    ax: Optional[Axes] = None,
):
    _ax = ax
    if not ax:
        fig, _ax = plt.subplots()
    neural_responses_sorted = neural_responses[sort_index]
    _ax.imshow(neural_responses_sorted, cmap=cmap)
    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    _ax.set_title(title)
    if not ax:
        plt.show()


def plot_orientation_fisher_information(
    stimulus: Stimulus1D,
    fisher_info: np.ndarray,
    /,
    colored_by_spatial_loc: bool = True,
    cmap: str = "inferno",
    ax: Optional[Axes] = None,
):
    _ax = ax
    if not ax:
        fig, _ax = plt.subplots()
    if colored_by_spatial_loc:
        _ax.scatter(
            stimulus.orientation, fisher_info, c=stimulus.spatial_loc, cmap=cmap
        )
    else:
        _ax.scatter(stimulus.orientation, fisher_info)
    _ax.set_title("Fisher Information $J(\\theta)$")
    _ax.set_xlabel("Orientation")
    _ax.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\pi /2$", "$-\pi /4$", "0", "$\pi/4$", "$\pi/2$"],
    )
    _ax.set_ylim(0, np.max(fisher_info * 1.1))
    if not ax:
        plt.show()


def plot_mds(
    data: np.ndarray,
    dim: int = 3,
    /,
    alpha: float = 0.3,
    c: Optional[np.ndarray] = None,
    cmap: str = "hsv",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    zlabel: str = "Dimension 3",
    title: str = "MDS Embedding of the neural responses (3D)",
    ax: Optional[Axes | Axes3D] = None,
):
    _ax = ax
    if not ax:
        fig = plt.figure()
        _ax = fig.add_subplot(111, projection="3d")
    embedding = MDS(n_components=dim)
    _transformed_3d = embedding.fit_transform(data)
    if dim == 2:
        _ax.scatter(
            _transformed_3d[:, 0],
            _transformed_3d[:, 1],
            c=c,
            alpha=alpha,
            cmap=cmap,
        )
    if dim == 3:
        if not isinstance(_ax, Axes3D):
            raise ValueError(
                "Incongruent dimension of the plot and matplotlib ax projection"
            )
        _ax.scatter(
            _transformed_3d[:, 0],
            _transformed_3d[:, 1],
            _transformed_3d[:, 2],
            c=c,
            alpha=alpha,
            cmap=cmap,
        )
    else:
        raise NotImplementedError(
            "Dimension for plot is not correct; need to be 2 or 3"
        )

    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    if dim == 3:
        _ax.set_zlabel(zlabel)
    _ax.set_title(title)
    if not ax:
        plt.show()


def plot_RDM(
    neural_responses: np.ndarray,
    sort_index: np.ndarray,
    /,
    cmap: str = "binary",
    ax: Optional[Axes] = None,
):
    _ax = ax
    if not ax:
        fig, _ax = plt.subplots()
    neural_responses_sorted = neural_responses[sort_index]

    p_dist = pdist(neural_responses_sorted)
    dist_mat = squareform(p_dist)

    _ax.imshow(dist_mat, cmap=cmap)
    if not ax:
        plt.show()


def plot_representational_distance(
    stimulus_value: np.ndarray,
    neural_responses: np.ndarray,
    /,
    c: Optional[np.ndarray] = None,
    alpha: float = 0.2,
    ax: Optional[Axes] = None,
):
    _ax = ax
    if not _ax:
        fig, _ax = plt.subplots()
    if c:
        assert len(c) == len(stimulus_value)
    sort_index = np.argsort(stimulus_value)
    neural_responses_sorted = neural_responses[sort_index]
    stim_sorted = stimulus_value[sort_index]
    stim_dist = pdist(stim_sorted.reshape(-1, 1))
    rep_dist = pdist(neural_responses_sorted)
    _ax.scatter(stim_dist, rep_dist, alpha=alpha, c=c)
    if not _ax:
        plt.show()


def plot_pca_scree(
    neural_responses: np.ndarray,
    dim: int = 15,
    /,
    ax: Optional[Axes] = None,
    title: str = "Scree Plot",
):
    _ax = ax
    if not _ax:
        fig, _ax = plt.subplots()

    pca = PCA()
    pca.fit(neural_responses)
    var = pca.explained_variance_ratio_
    _ax.scatter(np.arange(dim) + 1, var[:dim])
    _ax.set_title(title)
    if not _ax:
        plt.show()
