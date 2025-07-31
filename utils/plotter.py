from typing import Optional, cast, Any
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
from typing import overload
from matplotlib.figure import Figure, SubFigure
import numpy.typing as npt

plt.rcParams["figure.figsize"] = [10.0, 10.0]
plt.rcParams["figure.dpi"] = 200


def get_max_vonmises(kappa: float | int):
    return np.exp(kappa) / (2 * np.pi * i0(kappa))


def _get_ax(
    ax: Optional[Axes | Axes3D] = None,
    fig: Optional[Figure | SubFigure] = None,
    projection: Optional[str] = None,
) -> Axes | Axes3D:
    if (ax is None) and (fig is None):
        return plt.subplot(projection=projection)
    elif isinstance(ax, (Axes, Axes3D)) and isinstance(fig, (Figure, SubFigure)):
        assert (
            ax in fig.get_axes()
        ), f"{ax} object is not in {fig}, cannot plot in the same figure"
        return cast(Axes, ax)
    elif isinstance(fig, (Figure, SubFigure)):
        fig = cast(Figure | SubFigure, fig)
        return fig.add_subplot(111, projection=projection)
    else:
        if isinstance(ax, Axes3D):
            return cast(Axes3D, ax)
        elif isinstance(ax, Axes):
            return cast(Axes, ax)
        raise ValueError("Invalid ax type")


@overload
def plot_neural_orientation_tuning_profile(
    stimulus: Stimulus1D,
    neuron_tuning_loc: npt.NDArray[np.floating[Any]],
    neuron_tuning_kappa: npt.NDArray[np.floating[Any]],
    neuron_tuning_amp: npt.NDArray[np.floating[Any]],
    *,
    plot_every: int = 300,
    alpha: float = 0.2,
    cmap: str = "twilight",
    ax: Axes,
): ...


@overload
def plot_neural_orientation_tuning_profile(
    stimulus: Stimulus1D,
    neuron_tuning_loc: npt.NDArray[np.floating[Any]],
    neuron_tuning_kappa: npt.NDArray[np.floating[Any]],
    neuron_tuning_amp: npt.NDArray[np.floating[Any]],
    *,
    plot_every: int = 300,
    alpha: float = 0.2,
    cmap: str = "twilight",
    fig: Figure | SubFigure,
): ...


@overload
def plot_neural_orientation_tuning_profile(
    stimulus: Stimulus1D,
    neuron_tuning_loc: npt.NDArray[np.floating[Any]],
    neuron_tuning_kappa: npt.NDArray[np.floating[Any]],
    neuron_tuning_amp: npt.NDArray[np.floating[Any]],
    *,
    plot_every: int = 300,
    alpha: float = 0.2,
    cmap: str = "twilight",
): ...


def plot_neural_orientation_tuning_profile(
    stimulus: Stimulus1D,
    neuron_tuning_loc: npt.NDArray[np.floating[Any]],
    neuron_tuning_kappa: npt.NDArray[np.floating[Any]],
    neuron_tuning_amp: npt.NDArray[np.floating[Any]],
    *,
    plot_every: int = 300,
    alpha: float = 0.2,
    cmap: str = "twilight",
    ax: Optional[Axes] = None,
    fig: Optional[Figure | SubFigure] = None,
):
    _ax = _get_ax(ax, fig)
    _ax.set_box_aspect(1)
    probe_stim = np.sort(stimulus.stimulus_orientation)
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
    _ax.set_title("Neural Tuning Function", wrap=True)
    _ax.set_xlabel("Orientation")
    _ax.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\pi /2$", "$-\pi /4$", "0", "$\pi/4$", "$\pi/2$"],
    )
    _ax.set_xlim(-np.pi, np.pi)
    _ax.set_ylim(
        0, np.max(get_max_vonmises(np.max(neuron_tuning_kappa[::plot_every])) * 1.1)
    )
    if (ax is None) and (fig is None):
        plt.show()


@overload
def plot_orientation_activation(
    neural_responses: npt.NDArray[np.floating[Any]],
    sort_index: npt.NDArray[np.int64],
    *,
    cmap: str = "binary",
    xlabel: str = "neuron_id",
    ylabel: str = "stimulus",
    title: str = "Neural Activation vs Stimulus",
    ax: Axes,
): ...


@overload
def plot_orientation_activation(
    neural_responses: npt.NDArray[np.floating[Any]],
    sort_index: npt.NDArray[np.int64],
    *,
    cmap: str = "binary",
    xlabel: str = "neuron_id",
    ylabel: str = "stimulus",
    title: str = "Neural Activation vs Stimulus",
    fig: Figure | SubFigure,
): ...


@overload
def plot_orientation_activation(
    neural_responses: npt.NDArray[np.floating[Any]],
    sort_index: npt.NDArray[np.int64],
    *,
    cmap: str = "binary",
    xlabel: str = "neuron_id",
    ylabel: str = "stimulus",
    title: str = "Neural Activation vs Stimulus",
): ...


def plot_orientation_activation(
    neural_responses: npt.NDArray[np.floating[Any]],
    sort_index: npt.NDArray[np.int64],
    *,
    cmap: str = "binary",
    xlabel: str = "neuron_id",
    ylabel: str = "stimulus",
    title: str = "Neural Activation vs Stimulus",
    ax: Optional[Axes] = None,
    fig: Optional[Figure | SubFigure] = None,
):
    _ax = _get_ax(ax, fig)
    _ax.set_box_aspect(1)
    neural_responses_sorted = neural_responses[sort_index]
    _ax.imshow(neural_responses_sorted, aspect="auto", cmap=cmap, origin="lower")
    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    _ax.set_title(title, wrap=True)
    l = len(neural_responses)
    _ax.set_yticks(
        [0, l / 4, l / 2, 3 * l / 4, l],
        ["$-\pi /2$", "$-\pi /4$", "0", "$\pi/4$", "$\pi/2$"],
    )
    if (ax is None) and (fig is None):
        plt.show()


@overload
def plot_orientation_fisher_information(
    stimulus: Stimulus1D,
    fisher_info: npt.NDArray[np.floating[Any]],
    *,
    colored_by_spatial_loc: bool = True,
    cmap: str = "inferno",
): ...


@overload
def plot_orientation_fisher_information(
    stimulus: Stimulus1D,
    fisher_info: npt.NDArray[np.floating[Any]],
    *,
    colored_by_spatial_loc: bool = True,
    cmap: str = "inferno",
    ax: Axes,
): ...
@overload
def plot_orientation_fisher_information(
    stimulus: Stimulus1D,
    fisher_info: npt.NDArray[np.floating[Any]],
    *,
    colored_by_spatial_loc: bool = True,
    cmap: str = "inferno",
    fig: Figure | SubFigure,
): ...
def plot_orientation_fisher_information(
    stimulus: Stimulus1D,
    fisher_info: npt.NDArray[np.floating[Any]],
    *,
    colored_by_spatial_loc: bool = True,
    cmap: str = "inferno",
    ax: Optional[Axes] = None,
    fig: Optional[Figure | SubFigure] = None,
):
    _ax = _get_ax(ax, fig)
    _ax.set_box_aspect(1)
    if colored_by_spatial_loc:
        _ax.scatter(
            stimulus.stimulus_orientation,
            fisher_info,
            c=stimulus.stimulus_location,
            cmap=cmap,
        )
    else:
        _ax.scatter(stimulus.stimulus_orientation, fisher_info)
    _ax.set_title("Fisher Information $J(\\theta)$", wrap=True)
    _ax.set_xlabel("Orientation")
    _ax.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\pi /2$", "$-\pi /4$", "0", "$\pi/4$", "$\pi/2$"],
    )
    _ax.set_ylim(0, np.max(fisher_info * 1.1))
    if (ax is None) and (fig is None):
        plt.show()


@overload
def plot_mds(
    data: npt.NDArray[np.floating[Any]],
    dim: int = 3,
    *,
    alpha: float = 0.3,
    c: Optional[np.ndarray] = None,
    cmap: str = "hsv",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    zlabel: str = "Dimension 3",
    grid: bool = True,
    show_axis: bool = True,
    title: str = "MDS Embedding of the neural responses (3D)",
): ...


@overload
def plot_mds(
    data: npt.NDArray[np.floating[Any]],
    dim: int = 3,
    *,
    alpha: float = 0.3,
    c: Optional[np.ndarray] = None,
    cmap: str = "hsv",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    zlabel: str = "Dimension 3",
    grid: bool = True,
    show_axis: bool = True,
    title: str = "MDS Embedding of the neural responses (3D)",
    ax: Axes | Axes3D,
): ...


@overload
def plot_mds(
    data: npt.NDArray[np.floating[Any]],
    dim: int = 3,
    *,
    alpha: float = 0.3,
    c: Optional[np.ndarray] = None,
    cmap: str = "hsv",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    zlabel: str = "Dimension 3",
    grid: bool = True,
    show_axis: bool = True,
    title: str = "MDS Embedding of the neural responses (3D)",
    fig: Figure | SubFigure,
): ...


def plot_mds(
    data: npt.NDArray[np.floating[Any]],
    dim: int = 3,
    *,
    alpha: float = 0.3,
    c: Optional[np.ndarray] = None,
    cmap: str = "hsv",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    zlabel: str = "Dimension 3",
    title: str = "MDS Embedding of the neural responses (3D)",
    grid: bool = True,
    show_axis: bool = True,
    ax: Optional[Axes | Axes3D] = None,
    fig: Optional[Figure | SubFigure] = None,
):
    if dim == 3:
        _ax = _get_ax(ax, fig, projection="3d")
        _ax.set_box_aspect((1, 1, 1))
    elif dim == 2:
        _ax = _get_ax(ax, fig)
        _ax.set_box_aspect(1)
    else:
        raise NotImplementedError(
            "Dimension for plot is not correct; need to be 2 or 3"
        )
    embedding = MDS(n_components=dim)
    _transformed = cast(npt.NDArray[np.floating[Any]], embedding.fit_transform(data))
    if dim == 2:
        _ax.scatter(
            _transformed[:, 0],
            _transformed[:, 1],
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
            _transformed[:, 0],
            _transformed[:, 1],
            _transformed[:, 2],
            c=c,
            alpha=alpha,
            cmap=cmap,
        )

    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    if dim == 3:
        _ax.set_zlabel(zlabel)
    _ax.set_title(title, wrap=True)
    _ax.grid(grid)
    if not show_axis:
        _ax.set_axis_off()
    if (ax is None) and (fig is None):
        plt.show()


@overload
def plot_RDM(
    neural_responses: npt.NDArray[np.floating[Any]],
    sort_index: npt.NDArray[np.int64],
    *,
    cmap: str = "binary",
): ...
@overload
def plot_RDM(
    neural_responses: npt.NDArray[np.floating[Any]],
    sort_index: npt.NDArray[np.int64],
    *,
    cmap: str = "binary",
    ax: Axes,
): ...
@overload
def plot_RDM(
    neural_responses: npt.NDArray[np.floating[Any]],
    sort_index: npt.NDArray[np.int64],
    *,
    cmap: str = "binary",
    fig: Figure | SubFigure,
): ...


def plot_RDM(
    neural_responses: npt.NDArray[np.floating[Any]],
    sort_index: npt.NDArray[np.int64],
    *,
    cmap: str = "binary",
    ax: Optional[Axes] = None,
    fig: Optional[Figure | SubFigure] = None,
):
    _ax = _get_ax(ax, fig)
    _ax.set_box_aspect(1)
    neural_responses_sorted = neural_responses[sort_index]

    p_dist = pdist(neural_responses_sorted)
    dist_mat = squareform(p_dist)

    _ax.imshow(dist_mat, aspect="auto", norm="linear", cmap=cmap)
    if (ax is None) and (fig is None):
        plt.show()


@overload
def plot_representational_distance(
    stimulus_value: npt.NDArray[np.floating[Any]],
    neural_responses: npt.NDArray[np.floating[Any]],
    *,
    c: Optional[np.ndarray] = None,
    alpha: float = 0.2,
    xlabel: str = "feature distance",
    ylabel: str = "representation distance",
    title: str = "Representation distance vs. Feature distance",
): ...


@overload
def plot_representational_distance(
    stimulus_value: npt.NDArray[np.floating[Any]],
    neural_responses: npt.NDArray[np.floating[Any]],
    *,
    c: Optional[np.ndarray] = None,
    alpha: float = 0.2,
    xlabel: str = "feature distance",
    ylabel: str = "representation distance",
    title: str = "Representation distance vs. Feature distance",
    ax: Axes,
): ...


@overload
def plot_representational_distance(
    stimulus_value: npt.NDArray[np.floating[Any]],
    neural_responses: npt.NDArray[np.floating[Any]],
    *,
    c: Optional[np.ndarray] = None,
    alpha: float = 0.2,
    xlabel: str = "feature distance",
    ylabel: str = "representation distance",
    title: str = "Representation distance vs. Feature distance",
    fig: Figure | SubFigure,
): ...


def plot_representational_distance(
    stimulus_value: npt.NDArray[np.floating[Any]],
    neural_responses: npt.NDArray[np.floating[Any]],
    *,
    c: Optional[np.ndarray] = None,
    alpha: float = 0.2,
    xlabel: str = "feature distance",
    ylabel: str = "representation distance",
    title: str = "Representation distance vs. Feature distance",
    ax: Optional[Axes] = None,
    fig: Optional[Figure | SubFigure] = None,
):
    _ax = _get_ax(ax, fig)
    _ax.set_box_aspect(1)
    if c:
        assert len(c) == len(stimulus_value)
    sort_index = np.argsort(stimulus_value)
    neural_responses_sorted = neural_responses[sort_index]
    stim_sorted = stimulus_value[sort_index]
    stim_dist = pdist(stim_sorted.reshape(-1, 1))
    rep_dist = pdist(neural_responses_sorted)
    _ax.scatter(stim_dist, rep_dist, alpha=alpha, c=c)
    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    _ax.set_title(title, wrap=True)
    if (ax is None) and (fig is None):
        plt.show()


@overload
def plot_pca_scree(
    neural_responses: npt.NDArray[np.floating[Any]],
    dim: int = 15,
    *,
    title: str = "Scree Plot",
): ...
@overload
def plot_pca_scree(
    neural_responses: npt.NDArray[np.floating[Any]],
    dim: int = 15,
    *,
    title: str = "Scree Plot",
    ax: Axes,
): ...
@overload
def plot_pca_scree(
    neural_responses: npt.NDArray[np.floating[Any]],
    dim: int = 15,
    *,
    title: str = "Scree Plot",
    fig: Figure | SubFigure,
): ...


def plot_pca_scree(
    neural_responses: npt.NDArray[np.floating[Any]],
    dim: int = 15,
    *,
    title: str = "Scree Plot",
    ax: Optional[Axes] = None,
    fig: Optional[Figure | SubFigure] = None,
):
    _ax = _get_ax(ax, fig)
    _ax.set_box_aspect(1)
    pca = PCA()
    pca.fit(neural_responses)
    var = pca.explained_variance_ratio_
    _use_dim = min(dim, len(var))
    _ax.scatter(np.arange(_use_dim) + 1, var[:_use_dim])
    _ax.set_title(title, wrap=True)
    _ax.set_ylim(0, 0.7)
    if (ax is None) and (fig is None):
        plt.show()
