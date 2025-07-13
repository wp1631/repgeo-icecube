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

# ============Define Parameters==============
NR_NUM = 3000

# Neuron Orientation Tuning
NR_OT_LOC_MIN = -np.pi  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = np.pi
NR_OT_KAPPA = 1
NR_LOC_W = 100

# Stimulus sample
ST_NUM = 1000
ST_OR_MIN = -np.pi  # Stimulus orientation
ST_OR_MAX = np.pi
ST_LOC_MIN = -3
ST_LOC_MAX = 3

# Channel paraneters
CH_NUM = 6
CH_OR_LOC_MIN = 0
CH_OR_LOC_MAX = 2 * np.pi
CH_OR_KAPPA = 5

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


# plot neural tuning profile
def plot_neural_orientation_tuning_profile(
    stimulus: Stimulus1D,
    neuron_tuning_loc: np.ndarray,
    neuron_tuning_kappa: np.ndarray,
    neuron_tuning_amp: np.ndarray,
    /,
    dpi: int = 200,
    plot_every: int = 300,
    alpha: float = 0.3,
    cmap: str = "viridis",
):
    fig, ax = plt.subplots(dpi=dpi)
    probe_stim = np.sort(stimulus.orientation)
    num_lines = 1 + len(neuron_tuning_loc) // plot_every
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


def plot_fisher_information():
    plt.scatter(stimulus.orientation, fisher_info)
    plt.title("Fisher Information $J( \\theta )$")
    plt.xlabel("Orientation")
    plt.xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\pi /2$", "$-\pi /4$", "0", "$\pi/4$", "$\pi/2$"],
    )
    plt.ylim(0, np.max(fisher_info * 1.4))
    plt.show()


plot_fisher_information()
#

embedding = MDS(n_components=3, n_init=1)
response_transformed_3d = embedding.fit_transform(neural_responses)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    response_transformed_3d[:, 0],
    response_transformed_3d[:, 1],
    response_transformed_3d[:, 2],
    c=stimulus_ori,
    alpha=0.3,
    cmap="hsv",
)

ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
plt.title("MDS Embedding of Sparse Orientation Coding, Neural Response MDS (3D)")
plt.show()

neural_responses_sorted = neural_responses[np.argsort(stimulus_ori)]

p_dist = pdist(neural_responses_sorted)
dist_mat = squareform(p_dist)

plt.imshow(dist_mat)
plt.colorbar()
plt.show()
