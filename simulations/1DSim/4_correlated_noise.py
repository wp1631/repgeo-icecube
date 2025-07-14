import numpy as np
from scipy.linalg import lstsq
from scipy.stats import special_ortho_group, vonmises
from scipy.special import i0
import matplotlib.pyplot as plt
from icecream import ic
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from utils.generators.classes_1D import NeuronArray1D, Stimulus1D
import matplotlib as mpl
from sklearn.decomposition import PCA

# ============Define Parameters==============
NR_NUM = 3000

# Neuron Orientation Tuning
NR_OT_LOC_MIN = -np.pi  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = np.pi
NR_OT_KAPPA = 10
NR_LOC_W = 100

# Stimulus sample
ST_NUM = 1000
ST_OR_MIN = -np.pi  # Stimulus orientation
ST_OR_MAX = np.pi
ST_LOC_MIN = -3
ST_LOC_MAX = 3

# Channel paraneters
CH_NUM = 6
CH_OR_LOC_MIN = -np.pi
CH_OR_LOC_MAX = np.pi
CH_OR_KAPPA = 3
CH_RECF_WIDTH = 100
CH_RECF_MIN = -3
CH_RECF_MAX = 3

# ============Data Generation==============

# initialize the neuron values
stimulus_ori = np.random.uniform(ST_OR_MIN, ST_OR_MAX, ST_NUM)
stimulus_loc = np.random.uniform(ST_LOC_MIN, ST_LOC_MAX, ST_NUM)

neuron_tuning_loc = np.linspace(NR_OT_LOC_MIN, NR_OT_LOC_MAX, NR_NUM)
# neuron_tuning_loc = np.random.uniform(NR_OT_LOC_MIN, NR_OT_LOC_MAX, NR_NUM)
neuron_tuning_kappa = np.full(NR_NUM, NR_OT_KAPPA)
neuron_tuning_amp = np.full(NR_NUM, 1)
neuron_recf_loc = np.random.uniform(ST_LOC_MIN, ST_LOC_MAX, NR_NUM)
neuron_recf_width = np.full(NR_NUM, NR_LOC_W)
stimulus = Stimulus1D(spatial_loc=stimulus_loc, orientation=stimulus_ori)


neuron_arr = NeuronArray1D(
    neuron_tuning_loc,
    neuron_tuning_kappa,
    neuron_tuning_amp,
    neuron_recf_loc,
    neuron_recf_width,
)


def get_max_vonmises(kappa: float):
    return np.exp(kappa) / (2 * np.pi * i0(kappa))


# ============Visualization==============


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
):
    fig, ax = plt.subplots(dpi=dpi)
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
        ax.plot(
            probe_stim,
            tuning_amp * vonmises.pdf(probe_stim, loc=tuning_loc, kappa=tuning_kappa),
            alpha=alpha,
            c=col,
        )
    ax.plot(
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
    plt.title("Neural Tuning Function")
    plt.xlabel("Orientation")
    plt.xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\pi /2$", "$-\pi /4$", "0", "$\pi/4$", "$\pi/2$"],
    )
    plt.ylim(
        0, np.max(get_max_vonmises(np.max(neuron_tuning_kappa[::plot_every])) * 1.1)
    )
    plt.show()


plot_neural_orientation_tuning_profile(
    stimulus, neuron_tuning_loc, neuron_tuning_kappa, neuron_tuning_amp
)

# get neural responses
neural_responses = neuron_arr.get_responses(stimulus)
neural_responses = np.array(neural_responses).T

deriv = neuron_arr.get_derivatives(stimulus)
deriv_normed = np.linalg.norm(deriv, axis=1)
fisher_info = deriv_normed / np.sqrt(NR_NUM)


def plot_orientation_activation(neural_responses, sort_index):
    neural_responses_sorted = neural_responses[sort_index]
    plt.imshow(neural_responses_sorted, cmap="binary")
    plt.xlabel("neuron id")
    plt.ylabel("stimulus")
    plt.colorbar()
    plt.show()


plot_orientation_activation(neural_responses, np.argsort(stimulus.orientation))


def plot_orientation_fisher_information(
    stimulus: Stimulus1D,
    fisher_info: np.ndarray,
    colored_by_spatial_loc: bool = True,
    cmap: str = "inferno",
):
    if colored_by_spatial_loc:
        plt.scatter(
            stimulus.orientation, fisher_info, c=stimulus.spatial_loc, cmap=cmap
        )
    else:
        plt.scatter(stimulus.orientation, fisher_info)
    plt.title("Fisher Information $J(\\theta)$")
    plt.xlabel("Orientation")
    plt.xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\pi /2$", "$-\pi /4$", "0", "$\pi/4$", "$\pi/2$"],
    )
    plt.ylim(0, np.max(fisher_info * 1.1))
    plt.show()


plot_orientation_fisher_information(stimulus, fisher_info)


def plot_mds(
    data: np.ndarray,
    dim: int = 3,
    alpha: float = 0.3,
    cmap: str = "hsv",
    /,
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    zlabel: str = "Dimension 3",
    title: str = "MDS Embedding of the neural responses (3D)",
):
    embedding = MDS(n_components=dim)
    _transformed_3d = embedding.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if dim == 2:
        ax.scatter(
            _transformed_3d[:, 0],
            _transformed_3d[:, 1],
            c=stimulus_ori,
            alpha=alpha,
            cmap=cmap,
        )
    if dim == 3:
        ax.scatter(
            _transformed_3d[:, 0],
            _transformed_3d[:, 1],
            _transformed_3d[:, 2],
            c=stimulus_ori,
            alpha=alpha,
            cmap=cmap,
        )
    else:
        raise NotImplementedError(
            "Dimension for plot is not correct; need to be 2 or 3"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if dim == 3:
        ax.set_zlabel(zlabel)
    plt.title(title)
    plt.show()


# Plot MDS of the neural responses
plot_mds(neural_responses)

# Plot MDS of the orientation gradient
plot_mds(deriv, title="MDS Embedding of the Gradient (3D)")

# TODO Plot MDS of the location gradient
# plot_mds()
sort_index = np.argsort(stimulus_ori)


def plot_RDM(neural_responses: np.ndarray, sort_index: np.ndarray, cmap="binary"):
    neural_responses_sorted = neural_responses[sort_index]

    p_dist = pdist(neural_responses_sorted)
    dist_mat = squareform(p_dist)

    plt.imshow(dist_mat, cmap=cmap)
    plt.colorbar()
    plt.show()


plot_RDM(neural_responses, sort_index)


def plot_representational_distance(
    stimulus_value: np.ndarray, neural_responses: np.ndarray
):
    sort_index = np.argsort(stimulus_value)
    neural_responses_sorted = neural_responses[sort_index]
    stim_sorted = stimulus_value[sort_index]
    stim_dist = pdist(stim_sorted.reshape(-1, 1))
    rep_dist = pdist(neural_responses_sorted)
    plt.scatter(stim_dist, rep_dist)
    plt.show()


plot_representational_distance(stimulus.orientation, neural_responses)


def PCA_scree_plot(neural_responses, dim: int = 15):
    pca = PCA()
    pca.fit(neural_responses)
    vars = pca.explained_variance_ratio_
    plt.scatter(np.arange(dim) + 1, vars[:dim])
    plt.show()


PCA_scree_plot(neural_responses)


# ============Measurement Simulation===============

MEASUREMENT_GRID_SIZE = 0.05


def create_voxel_sampling(
    neural_responses: np.ndarray,
    neuron_loc: np.ndarray,
    voxel_width: float = MEASUREMENT_GRID_SIZE,
    min_loc: float = -3,
    max_loc: float = 3,
    statistic: str = "mean",
):
    stats_dict = {
        "mean": np.mean,
        "median": np.median,
    }
    stat_func = stats_dict.get(statistic, np.mean)
    voxel_bounds = np.arange(min_loc, max_loc, voxel_width)
    all_channel = []
    for i in range(len(voxel_bounds)):
        # get the index for bounded location
        use_index = np.logical_and(
            neuron_loc >= voxel_bounds[i], neuron_loc <= voxel_bounds[i] + voxel_width
        )
        voxel_responses = stat_func(neural_responses[:, use_index], axis=1).reshape(
            -1, 1
        )
        all_channel.append(voxel_responses)
    measurement = np.hstack(all_channel)
    return measurement


measurement = create_voxel_sampling(
    neural_responses, neuron_recf_loc, MEASUREMENT_GRID_SIZE
)

plot_RDM(measurement, sort_index)
# ============Inverted Encoding Model==============

## Channel Responses
channel_tuning_loc = np.linspace(CH_OR_LOC_MIN, CH_OR_LOC_MAX, CH_NUM)
channel_tuning_kappa = np.full(CH_NUM, CH_OR_KAPPA)
channel_tuning_amp = np.full(CH_NUM, 1)
channel_recf_loc = np.random.uniform(CH_RECF_MIN, CH_RECF_MAX, CH_NUM)
channel_recf_width = np.full(CH_NUM, CH_RECF_WIDTH)

channel_arr = NeuronArray1D(
    tuning_loc=channel_tuning_loc,
    tuning_kappa=channel_tuning_kappa,
    tuning_amplitude=channel_tuning_amp,
    recf_loc=channel_recf_loc,
    recf_width=channel_recf_width,
)
channel_activation = channel_arr.get_responses(stimulus=stimulus).T

## Get Weighting Variable
fit_res = lstsq(channel_activation, measurement)
mapping_weight = fit_res[0]
residues = fit_res[1]
ic(residues.shape)
total_res = np.sum(residues)
ic(total_res.shape)


## Get total variance
def get_total_variance(signals: np.ndarray):
    pca = PCA()
    pca.fit(signals)
    return np.sum(pca.explained_variance_)


total_var = get_total_variance(signals=channel_activation)
ic(total_var.shape)
var_ratio = total_var / total_res
ic(var_ratio.shape)
ic(var_ratio)

inv_weight = np.linalg.pinv(mapping_weight)

## Test the reconstruction
reconstructed_channal_responses = measurement @ inv_weight
plot_mds(reconstructed_channal_responses, title="Reconstructed Cahnnel Responses")
# ================Covariate Noise==================
