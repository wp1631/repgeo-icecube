import numpy as np
from icecream import ic
from numpy.ma import argsort
from utils.generators.classes_1D import NeuronArray1D, Stimulus1D
from utils.generators.noise import create_block_noise
from utils.statistical_sampling import create_voxel_sampling
from utils.iem import IEM1D
from utils.rep_metrics import global_distance_variance, global_neigbor_dice, linear_CKA
import matplotlib.pyplot as plt
from utils.plotter import (
    plot_neural_orientation_tuning_profile,
    plot_mds,
    plot_pca_scree,
    plot_RDM,
    plot_orientation_fisher_information,
    plot_orientation_activation,
)
import seaborn as sns
import pandas as pd

# ============Define Parameters==============
NR_NUM = 3000

# Neuron Orientation Tuning
NR_OT_LOC_MIN = -np.pi  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = np.pi
NR_OT_KAPPA = 5
NR_LOC_W = 0.03

# Stimulus sample
ST_NUM = 1000
ST_OR_MIN = -np.pi  # stimulus.stimulus_orientation
ST_OR_MAX = np.pi
ST_LOC_MIN = -3
ST_LOC_MAX = 3

# Channel paraneters
CH_NUM = 12
CH_OR_LOC_MIN = -np.pi
CH_OR_LOC_MAX = np.pi
CH_OR_KAPPA = 3
CH_RECF_WIDTH = 1
CH_RECF_MIN = -3
CH_RECF_MAX = 3
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

# SHow sampling space
plt.scatter(neuron_recf_loc, neuron_tuning_loc, s=1, alpha=0.3)
plt.xlabel("Receptive Field Location")
plt.ylabel("Orientation Tuning Center")
plt.yticks(
    [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    ["$-\pi /2$", "$-\pi / 4$", "0", "$\pi / 4$", "$\pi / 2$"],
)
plt.show()


neuron_arr = NeuronArray1D(
    neuron_tuning_loc,
    neuron_tuning_kappa,
    neuron_tuning_amp,
    neuron_recf_loc,
    neuron_recf_width,
)

# get neural responses
neural_responses = neuron_arr.get_responses(stimulus)
neural_responses = np.array(neural_responses).T

deriv = neuron_arr.get_derivatives(stimulus)
deriv_normed = np.linalg.norm(deriv, axis=1)
fisher_info = deriv_normed / np.sqrt(NR_NUM)
sort_index = np.argsort(stimulus_ori)

plot_neural_orientation_tuning_profile(
    stimulus, neuron_tuning_loc, neuron_tuning_kappa, neuron_tuning_amp
)

MEASUREMENT_GRID_SIZE = 0.05

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

channel_arr = NeuronArray1D(
    tuning_loc=channel_tuning_loc,
    tuning_kappa=channel_tuning_kappa,
    tuning_amplitude=channel_tuning_amp,
    recf_loc=channel_recf_loc,
    recf_width=channel_recf_width,
)
channel_activation = channel_arr.get_responses(stimulus=stimulus).T

iem_obj = IEM1D(channel_arr)
iem_obj.fit(stimulus, measurement)


## Get Weighting Variable
mapping_weight = iem_obj.encode_weight
residues = iem_obj.encode_residues
ic(residues.shape)
total_res = np.sum(residues)
ic(total_res.shape)
inv_weight = iem_obj.decode_weight

NEURONAL_NOISE_AMPLITUDE = 0.05
MEASUREMENT_NOISE_AMPLITUDE = 0.1

base_signal_noise = np.random.normal(loc=0, scale=1, size=neural_responses.shape)
noisy_responses = neural_responses + NEURONAL_NOISE_AMPLITUDE * base_signal_noise

# responses
plot_RDM(neural_responses, sort_index)
plot_mds(neural_responses, c=stimulus.stimulus_orientation)

base_measurement_noise = np.random.normal(loc=0, scale=1, size=measurement.shape)
noisy_measurement = measurement + MEASUREMENT_NOISE_AMPLITUDE * base_measurement_noise

noisy_iem = IEM1D(channel_arr)
noisy_iem.fit(stimulus, noisy_measurement)

noisy_reconstruct_channel = noisy_iem.decode(noisy_measurement)

# ================Covariate Noise==================
spatial_block_noise_response = neural_responses.copy()
block_noise = create_block_noise(
    block_size=200, total_size=NR_NUM, observation=ST_NUM, minor_amp=0.1, amplitude=0.2
)

spatial_block_noise_response[:, np.argsort(neuron_recf_loc)] += block_noise

# spatial block
neural_corr = pd.DataFrame(neural_responses).corr()
spatial_block_noise_corr = pd.DataFrame(block_noise).corr()
plt.imshow(block_noise)
plt.show()
spatial_res_block_noise_corr = pd.DataFrame(
    spatial_block_noise_response[:, argsort(neuron_recf_loc)]
).corr()
plt.imshow(spatial_res_block_noise_corr)
plt.show()
# spatial_block_noise_response
plot_RDM(spatial_block_noise_response, sort_index)
plot_mds(spatial_block_noise_response, c=stimulus.stimulus_orientation)

block_noise_measurment = create_voxel_sampling(
    spatial_block_noise_response, neuron_recf_loc
)

block_noise_iem = IEM1D(channel_arr)
block_noise_iem.fit(stimulus, block_noise_measurment)
block_noisy_channel = block_noise_iem.decode(block_noise_measurment)
plot_RDM(block_noisy_channel, sort_index)
plot_mds(block_noisy_channel, c=stimulus.stimulus_orientation)


# ic("measurement vs Neurons")
# ic(np.mean(global_neigbor_dice(measurement, spatial_block_noise_response)))
# ic(linear_CKA(measurement, neural_responses))
# ic(global_distance_variance(measurement, spatial_block_noise_response))
# ic("Recon vs Channel")
# ic(np.mean(global_neigbor_dice(block_noisy_channel, channel_activation)))
# ic(linear_CKA(block_noisy_channel, channel_activation))
# ic(global_distance_variance(block_noisy_channel, channel_activation))
# ic("Recon vs Neurons")
# ic(np.mean(global_neigbor_dice(block_noisy_channel, spatial_block_noise_response)))
# ic(linear_CKA(block_noisy_channel, neural_responses))
# ic(global_distance_variance(block_noisy_channel, spatial_block_noise_response))
