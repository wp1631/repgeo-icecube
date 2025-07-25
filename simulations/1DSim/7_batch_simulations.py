from typing import Tuple
import numpy as np
from itertools import product
from utils.generators.classes_1D import NeuronArray1D, Stimulus1D
from utils.generators.noise import create_block_noise
from utils.statistical_sampling import create_voxel_sampling
from utils.iem import IEM1D
from utils.rep_metrics import global_distance_variance, global_neigbor_dice, linear_CKA

# Neuron Orientation Tuning
NR_NUM = 3000
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

# Measurement
MEASUREMENT_GRID_SIZE = 0.05
BLOCK_NEURONAL_NOISE_AMPLITUDE = 0.05
BLOCK_NEURONAL_MINOR_NOISE_AMPLITUDE = 0.1
MEASUREMENT_NOISE_AMPLITUDE = 0.1

REPEAT_NUM = 100
# Varying parameters
## Neuron
SP_NR_NUM = np.logspace(1, 6, num=10, endpoint=True, dtype=np.int64)
SP_NR_OT_KAPPA = np.logspace(-4, 4, num=10, endpoint=True)
SP_NR_RECF_W = np.logspace(-5, 5, num=10, endpoint=True)
## Channel
SP_CH_NUM = np.linspace(1, 50, endpoint=True, num=50, dtype=np.int64)
SP_CH_OT_KAPPA = np.logspace(-4, 4, num=10, endpoint=True)
SP_CH_RECF_W = np.logspace(-5, 5, num=10, endpoint=True)

## MEASUREMENT
SP_MEASUREMENT_GRID_SIZE = np.linspace(
    0.01, ST_LOC_MAX - ST_LOC_MIN, endpoint=True, num=20
)
SP_BLOCK_NOISE_AMP = np.logspace(-2, 2, endpoint=True, base=10, num=5)
SP_BLOCK_NEURON_SIZE = np.logspace(0, 5, endpoint=True, base=10, num=10, dtype=np.int64)
SP_BLOCK_NOISE_MINOR_AMP = np.logspace(-2, 2, endpoint=True, base=10, num=5)
SP_MEASUREMNET_NOISE_AMP = np.logspace(-2, 2, endpoint=True, base=10, num=5)

# parameter space
NR_NUM_L = []
NR_OT_KAPPA_L = []
NR_RECF_W_L = []

CH_NUM_L = []
CH_OT_KP_L = []
CH_RECF_W_L = []

BLOCK_NOISE_AMP_L = []
BLOCK_NEURON_SIZE_L = []
BLOCK_NEURON_MINOR_AMP_L = []
MEASURE_NOISE_AMP_L = []

# cka
mr_lcka_list = []
mn_lcka_list = []
mc_lcka_list = []
rn_lcka_list = []
rc_lcka_list = []
nc_lcka_list = []
# mean global displacement
mr_mgd_list = []
mn_mgd_list = []
mc_mgd_list = []
rn_mgd_list = []
rc_mgd_list = []
nc_mgd_list = []
# mean neighbor dice index
mr_mnd_list = []
mn_mnd_list = []
mc_mnd_list = []
rn_mnd_list = []
rc_mnd_list = []
nc_mnd_list = []


def get_three_metrics(X: np.ndarray, Y: np.ndarray) -> Tuple:
    return (
        linear_CKA(X, Y),
        np.mean(global_neigbor_dice(X, Y)),
        global_distance_variance(X, Y),
    )


def generate_metrics(
    NR_NUM=3000,
    NR_OT_KAPPA=5,
    NR_LOC_W=0.03,
    ST_NUM=1000,
    CH_NUM=12,
    CH_OR_KAPPA=3,
    CH_RECF_WIDTH=1,
    CH_RECF_MIN=-3,
    CH_RECF_MAX=3,
    MEASUREMENT_GRID_SIZE=0.05,
    BLOCK_NEURONAL_NOISE_AMPLITUDE=0.05,
    NEURON_BLOCK_SIZE=200,
    BLOCK_NEURONAL_MINOR_NOISE_AMPLITUDE=0.1,
    MEASUREMENT_NOISE_AMPLITUDE=0.1,
):
    # initialize the neuron values
    stimulus_ori = np.random.uniform(ST_OR_MIN, ST_OR_MAX, ST_NUM)
    stimulus_loc = np.random.uniform(ST_LOC_MIN, ST_LOC_MAX, ST_NUM)
    stimulus_contrast = np.full_like(stimulus_loc, 1)
    stimulus_size = np.full_like(stimulus_loc, 1)

    neuron_tuning_loc = np.linspace(NR_OT_LOC_MIN, NR_OT_LOC_MAX, NR_NUM)
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

    # ================Covariate Noise==================
    spatial_block_noise_response = neural_responses.copy()
    spatial_block_noise_response[:, np.argsort(neuron_recf_loc)] += create_block_noise(
        block_size=NEURON_BLOCK_SIZE,
        total_size=NR_NUM,
        observation=ST_NUM,
        amplitude=BLOCK_NEURONAL_NOISE_AMPLITUDE,
        minor_amp=BLOCK_NEURONAL_MINOR_NOISE_AMPLITUDE,
    )

    block_noise_measurment = create_voxel_sampling(
        spatial_block_noise_response, neuron_recf_loc, voxel_width=MEASUREMENT_GRID_SIZE
    )
    block_noise_measurment += MEASUREMENT_NOISE_AMPLITUDE * np.random.normal(
        0, 1, block_noise_measurment.shape
    )

    block_noise_iem = IEM1D(channel_arr)
    block_noise_iem.fit(stimulus, block_noise_measurment)
    block_noisy_channel = block_noise_iem.decode(block_noise_measurment)

    # channel vs recon (cr)
    _ = get_three_metrics(channel_activation, block_noisy_channel)
    rc_lcka_list.append(_[0])
    rc_mnd_list.append(_[1])
    rc_mgd_list.append(_[2])

    # channel vs neuron (cn)
    _ = get_three_metrics(channel_activation, spatial_block_noise_response)
    nc_lcka_list.append(_[0])
    nc_mnd_list.append(_[1])
    nc_mgd_list.append(_[2])

    # recon vs neuron (rn)
    _ = get_three_metrics(block_noisy_channel, spatial_block_noise_response)
    rn_lcka_list.append(_[0])
    rn_mnd_list.append(_[1])
    rn_mgd_list.append(_[2])

    # measure vs channel (mc)
    _ = get_three_metrics(block_noise_measurment, channel_activation)
    mc_lcka_list.append(_[0])
    mc_mnd_list.append(_[1])
    mc_mgd_list.append(_[2])

    # measure vs recon (mr)
    _ = get_three_metrics(block_noise_measurment, block_noisy_channel)
    mr_lcka_list.append(_[0])
    mr_mnd_list.append(_[1])
    mr_mgd_list.append(_[2])

    # measurement vs neuron (mn)
    _ = get_three_metrics(block_noise_measurment, spatial_block_noise_response)
    mn_lcka_list.append(_[0])
    mn_mnd_list.append(_[1])
    mn_mgd_list.append(_[2])


if __name__ == "__main__":
    for (
        NR_NUM,
        NR_OT_KAPPA,
        NR_RECF_W,
        CH_NUM,
        CH_OT_KAPPA,
        CH_RECF_W,
        MEASUREMNT_GRID_SIZE,
        BLOCK_NOISE_AMP,
        BLOCK_NEURON_SIZE,
        BLOCK_NOISE_MINOR_AMP,
        MEASURE_NOISE_AMP,
    ) in product(
        SP_NR_NUM,
        SP_NR_OT_KAPPA,
        SP_NR_RECF_W,
        SP_CH_NUM,
        SP_CH_OT_KAPPA,
        SP_CH_RECF_W,
        SP_MEASUREMENT_GRID_SIZE,
        SP_BLOCK_NOISE_AMP,
        SP_BLOCK_NEURON_SIZE,
        SP_BLOCK_NOISE_MINOR_AMP,
        SP_MEASUREMNET_NOISE_AMP,
    ):
        pass

# import matplotlib.pyplot as plt

# ax = plt.subplot()
# for item in zip(
#     rc_lcka_list, nc_lcka_list, rn_lcka_list, mc_lcka_list, mr_lcka_list, mn_lcka_list
# ):
#     ax.plot(item, alpha=0.3)
# plt.title("CKA")
# plt.xticks(
#     [0, 1, 2, 3, 4, 5],
#     [
#         "recon/channel",
#         "channel/neuron",
#         "recon/neuron",
#         "measure/channel",
#         "measure_recon",
#         "measure/neuron",
#     ],
# )
# plt.show()
# ax = plt.subplot()
# for item in zip(
#     rc_mnd_list, nc_mnd_list, rn_mnd_list, mc_mnd_list, mr_mnd_list, mn_mnd_list
# ):
#     ax.plot(item, alpha=0.3)
# plt.title("Mean neighbor dice")
# plt.xticks(
#     [0, 1, 2, 3, 4, 5],
#     [
#         "recon/channel",
#         "channel/neuron",
#         "recon/neuron",
#         "measure/channel",
#         "measure_recon",
#         "measure/neuron",
#     ],
# )
# plt.show()
# ax = plt.subplot()
# for item in zip(
#     rc_mgd_list, nc_mgd_list, rn_mgd_list, mc_mgd_list, mr_mgd_list, mn_mgd_list
# ):
#     ax.plot(item, alpha=0.3)
# plt.title("Mean global displacement")
# plt.xticks(
#     [0, 1, 2, 3, 4, 5],
#     [
#         "recon/channel",
#         "channel/neuron",
#         "recon/neuron",
#         "measure/channel",
#         "measure_recon",
#         "measure/neuron",
#     ],
# )
# plt.show()
# # cka
#     mr_lcka_list = []
#     mn_lcka_list = []
#     mc_lcka_list = []
#     rn_lcka_list = []
#     rc_lcka_list = []
#     nc_lcka_list = []
# # mean global displacement
#     mr_mgd_list = []
#     mn_mgd_list = []
#     mc_mgd_list = []
#     rn_mgd_list = []
#     rc_mgd_list = []
#     nc_mgd_list = []
# # mean neighbor dice index
#     mr_mnd_list = []
#     mn_mnd_list = []
#     mc_mnd_list = []
#     rn_mnd_list = []
#     rc_mnd_list = []
#     nc_mnd_list = []
