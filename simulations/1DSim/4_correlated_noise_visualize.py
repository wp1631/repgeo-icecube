import numpy as np
from icecream import ic
from utils.generators.classes_1D import NeuronArray1D, Stimulus1D
from utils.generators.noise import create_block_noise
from sklearn.decomposition import PCA
from utils.plotter import (
    plot_orientation_fisher_information,
    plot_mds,
    plot_RDM,
    plot_orientation_activation,
    plot_representational_distance,
    plot_neural_orientation_tuning_profile,
    plot_pca_scree,
)
from utils.statistical_sampling import create_voxel_sampling
from utils.iem import IEM1D
import matplotlib.pyplot as plt
from utils.rep_metrics import global_distance_variance, global_neigbor_dice, linear_CKA
import matplotlib.pyplot as plt

# ============Define Parameters==============
NR_NUM = 3000

# Neuron Orientation Tuning
NR_OT_LOC_MIN = -np.pi  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = np.pi
NR_OT_KAPPA = 5
NR_LOC_W = 0.03

# Stimulus sample
ST_NUM = 500
ST_OR_MIN = -np.pi  # stimulus.stimulus_orientation
ST_OR_MAX = np.pi
ST_LOC_MIN = -10
ST_LOC_MAX = 10

# Channel paraneters
CH_NUM = 6
CH_OR_LOC_MIN = -np.pi
CH_OR_LOC_MAX = np.pi
CH_OR_KAPPA = 3
CH_RECF_WIDTH = 3
CH_RECF_MIN = ST_LOC_MIN
CH_RECF_MAX = ST_LOC_MAX

MEASUREMENT_GRID_SIZE = 0.05

NEURONAL_NOISE_AMPLITUDE = 0.05
BLOCK_MINOR_NOISE_AMPLITUDE = 0.3
MEASUREMENT_NOISE_AMPLITUDE = 0.02
# ============Data Generation==============

# initialize the neuron values
stimulus_ori = np.random.uniform(ST_OR_MIN, ST_OR_MAX, ST_NUM)
stimulus_loc = np.random.uniform(ST_LOC_MIN, ST_LOC_MAX, ST_NUM)
stimulus_contrast = np.full_like(stimulus_loc, 1)
stimulus_size = np.full_like(stimulus_loc, 1)


neuron_tuning_loc = np.linspace(NR_OT_LOC_MIN, NR_OT_LOC_MAX, NR_NUM)
# neuron_tuning_loc = np.random.uniform(NR_OT_LOC_MIN, NR_OT_LOC_MAX, NR_NUM)
neuron_tuning_kappa = np.full(NR_NUM, NR_OT_KAPPA)
neuron_tuning_amp = np.full(NR_NUM, 1)
neuron_recf_loc = np.random.uniform(ST_LOC_MIN, ST_LOC_MAX, NR_NUM)
neuron_recf_width = np.full(NR_NUM, NR_LOC_W)
stimulus = Stimulus1D(
    stimulus_location=stimulus_loc,
    stimulus_orientation=stimulus_ori,
    stimulus_contrast=stimulus_contrast,
    stimulus_size=stimulus_size,
)

plt.scatter(neuron_recf_loc, neuron_tuning_loc, s=2)
plt.title("Tuning vs Receptive field")
plt.show()


neuron_arr = NeuronArray1D(
    neuron_tuning_loc,
    neuron_tuning_kappa,
    neuron_tuning_amp,
    neuron_recf_loc,
    neuron_recf_width,
)


# ============Visualization==============


# get neural responses
neural_responses = neuron_arr.get_responses(stimulus)
neural_responses = np.array(neural_responses).T

deriv = neuron_arr.get_derivatives(stimulus)
deriv_normed = np.linalg.norm(deriv, axis=1)
fisher_info = deriv_normed / np.sqrt(NR_NUM)
sort_index = np.argsort(stimulus_ori)


def plot_all():
    neuron_fig = plt.figure(dpi=300)
    neuron_subfigs = neuron_fig.subfigures(
        1, 8, height_ratios=np.full(1, 1), width_ratios=np.full(8, 1)
    )
    plot_neural_orientation_tuning_profile(
        stimulus,
        neuron_tuning_loc,
        neuron_tuning_kappa,
        neuron_tuning_amp,
        plot_every=100,
        fig=neuron_subfigs[0],
    )

    plot_orientation_activation(
        neural_responses,
        np.argsort(stimulus.stimulus_orientation),
        fig=neuron_subfigs[1],
    )

    plot_orientation_fisher_information(stimulus, fisher_info, fig=neuron_subfigs[2])

    plot_mds(neural_responses, c=stimulus.stimulus_orientation, fig=neuron_subfigs[3])

    plot_mds(
        deriv,
        c=stimulus.stimulus_orientation,
        title="MDS Embedding of the Gradient (3D)",
        fig=neuron_subfigs[4],
    )

    plot_RDM(neural_responses, sort_index, fig=neuron_subfigs[5])

    plot_representational_distance(
        stimulus.stimulus_orientation, neural_responses, fig=neuron_subfigs[6]
    )

    plot_pca_scree(neural_responses, fig=neuron_subfigs[7])

    neuron_fig.show()


plot_neural_orientation_tuning_profile(
    stimulus,
    neuron_tuning_loc,
    neuron_tuning_kappa,
    neuron_tuning_amp,
    plot_every=100,
)

plot_orientation_activation(
    neural_responses,
    np.argsort(stimulus.stimulus_orientation),
)

plot_orientation_fisher_information(stimulus, fisher_info)

plot_mds(neural_responses, c=stimulus.stimulus_orientation)

# plot_mds(
#     deriv,
#     c=stimulus.stimulus_orientation,
#     title="MDS Embedding of the Gradient (3D)",
# )

plot_RDM(neural_responses, sort_index)

plot_representational_distance(stimulus.stimulus_orientation, neural_responses)

plot_pca_scree(neural_responses)


measurement = create_voxel_sampling(
    neural_responses, neuron_recf_loc, MEASUREMENT_GRID_SIZE
)
# ============Inverted Encoding Model==============

## Channel Responses
channel_tuning_loc = np.linspace(CH_OR_LOC_MIN, CH_OR_LOC_MAX, CH_NUM)
channel_tuning_kappa = np.full(CH_NUM, CH_OR_KAPPA)
channel_tuning_amp = np.full(CH_NUM, 1)
channel_recf_loc = np.random.uniform(CH_RECF_MIN, CH_RECF_MAX, CH_NUM)
channel_recf_width = np.full(CH_NUM, CH_RECF_WIDTH)

plt.scatter(channel_recf_loc, channel_tuning_loc, s=2)
plt.title("Channel Tuning vs Receptive field")
plt.show()

channel_arr = NeuronArray1D(
    tuning_loc=channel_tuning_loc,
    tuning_kappa=channel_tuning_kappa,
    tuning_amplitude=channel_tuning_amp,
    recf_loc=channel_recf_loc,
    recf_width=channel_recf_width,
)
channel_activation = channel_arr.get_responses(stimulus=stimulus).T

plot_mds(
    channel_activation,
    c=stimulus.stimulus_orientation,
    title="MDS Embedding of Channel Activation (3D)",
)
plot_RDM(channel_activation, sort_index)
plot_pca_scree(channel_activation, title="Scree plot channel activation")

iem_obj = IEM1D(channel_arr)
iem_obj.fit(stimulus, measurement)


## Get Weighting Variable
mapping_weight = iem_obj.encode_weight
residues = iem_obj.encode_residues
ic(residues.shape)
total_res = np.sum(residues)
ic(total_res.shape)
inv_weight = iem_obj.decode_weight


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


## Test the reconstruction
reconstructed_channal_responses = measurement @ inv_weight

plot_mds(
    reconstructed_channal_responses,
    c=stimulus.stimulus_orientation,
    title="Reconstructed Channel Responses",
)

plot_representational_distance(
    stimulus.stimulus_orientation,
    reconstructed_channal_responses,
    title="Reconstructed Representation Distance vs. Feature Distance",
)

## Noisy Encoding


base_signal_noise = np.random.normal(loc=0, scale=1, size=neural_responses.shape)
noisy_responses = neural_responses + NEURONAL_NOISE_AMPLITUDE * base_signal_noise

base_measurement_noise = np.random.normal(loc=0, scale=1, size=measurement.shape)
noisy_measurement = measurement + MEASUREMENT_NOISE_AMPLITUDE * base_measurement_noise

plot_mds(
    noisy_responses,
    c=stimulus.stimulus_orientation,
    title="MDS Embedding of Noisy Neural Responses (3D)",
)
plot_RDM(noisy_responses, sort_index)
plot_pca_scree(noisy_responses, title="Scree plot: Noisy Neural Responses Scree plot")

plot_mds(
    noisy_measurement,
    c=stimulus.stimulus_orientation,
    title="MDS Embedding of Noisy Measurement (3D)",
)
plot_RDM(noisy_measurement, sort_index)
plot_pca_scree(noisy_measurement, title="Scree plot: Noisy measurement")

noisy_iem = IEM1D(channel_arr)
noisy_iem.fit(stimulus, noisy_measurement)

noisy_reconstruct_channel = noisy_iem.decode(noisy_measurement)

plot_mds(
    noisy_reconstruct_channel,
    c=stimulus.stimulus_orientation,
    title="Noisy Reconstructed MDS (3D)",
)
plot_RDM(noisy_reconstruct_channel, sort_index)
plot_pca_scree(noisy_reconstruct_channel, title="Scree plot: noisy reconstruction")
# ================Covariate Noise==================

spatial_block_noise_response = neural_responses.copy()
spatial_block_noise_response[
    :, np.argsort(neuron_recf_loc)
] += MEASUREMENT_NOISE_AMPLITUDE * create_block_noise(
    block_size=200,
    total_size=NR_NUM,
    observation=ST_NUM,
    amplitude=1,
    minor_amp=BLOCK_MINOR_NOISE_AMPLITUDE,
)

plot_mds(
    spatial_block_noise_response,
    c=stimulus.stimulus_orientation,
    title="MDS Embedding of Block Noisy Neural Responses (3D)",
)
plot_RDM(
    spatial_block_noise_response,
    sort_index,
)
plot_pca_scree(
    spatial_block_noise_response, title="Scree plot: spatial block neuronal noise"
)

recf_block_noise_measure = create_voxel_sampling(
    spatial_block_noise_response, neuron_recf_loc, MEASUREMENT_GRID_SIZE
)
plot_mds(
    recf_block_noise_measure,
    c=stimulus.stimulus_orientation,
    title="MDS Embedding of Block Noisy Measurement (3D)",
)
plot_RDM(
    recf_block_noise_measure,
    sort_index,
)
plot_pca_scree(
    recf_block_noise_measure, title="Scree plot: fMRI block noise measurement"
)

block_noise_iem = IEM1D(channel_arr)
block_noise_iem.fit(stimulus, recf_block_noise_measure)
block_noisy_channel_recon = block_noise_iem.decode(recf_block_noise_measure)

plot_mds(
    block_noisy_channel_recon,
    c=stimulus.stimulus_orientation,
    title="MDS Embedding of Block Noisy Channel Reconstruction (3D)",
)
plot_RDM(
    block_noisy_channel_recon,
    sort_index,
)
plot_pca_scree(
    block_noisy_channel_recon,
    title="Scree plot: Channel reconstruction - fMRI block measurement",
)
