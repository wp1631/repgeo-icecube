import numpy as np
from utils.generators.classes_1D import NeuronArray1D, Stimulus1D
from utils.generators.noise import create_block_noise
from utils.statistical_sampling import create_voxel_sampling
from utils.iem import IEM1D
from utils.rep_metrics import global_distance_variance, global_neigbor_dice, linear_CKA
import seaborn as sns
import pandas as pd
from icecream import ic

TOTAL_SIMULATION = 30
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


# Stimulus sample
ST_NUM = 1000
ST_OR_MIN = -np.pi  # stimulus.stimulus_orientation
ST_OR_MAX = np.pi
ST_LOC_MIN = -3
ST_LOC_MAX = 3

# Neuron Orientation Tuning
NR_NUM = 3001
NR_OT_LOC_MIN = ST_OR_MIN  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = ST_OR_MAX
NR_OT_KAPPA = 10
NR_LOC_W = 0.03

# Channel paraneters
CH_NUM = 12
CH_OR_LOC_MIN = -np.pi
CH_OR_LOC_MAX = np.pi
CH_OR_KAPPA = 3
CH_RECF_WIDTH = 1
CH_RECF_MIN = ST_LOC_MIN
CH_RECF_MAX = ST_LOC_MAX

MEASUREMENT_GRID_SIZE = 0.05

for i in range(TOTAL_SIMULATION):
    ic(i)
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
    total_res = np.sum(residues)
    inv_weight = iem_obj.decode_weight

    NEURONAL_NOISE_AMPLITUDE = 0.2
    MEASUREMENT_NOISE_AMPLITUDE = 0.1

    base_measurement_noise = np.random.normal(loc=0, scale=1, size=measurement.shape)
    noisy_measurement = (
        measurement + MEASUREMENT_NOISE_AMPLITUDE * base_measurement_noise
    )

    noisy_iem = IEM1D(channel_arr)
    noisy_iem.fit(stimulus, noisy_measurement)

    noisy_reconstruct_channel = noisy_iem.decode(noisy_measurement)

    # ================Covariate Noise==================
    spatial_block_noise_response = neural_responses.copy()
    spatial_block_noise_response[
        :, np.argsort(neuron_recf_loc)
    ] += NEURONAL_NOISE_AMPLITUDE * create_block_noise(
        block_size=200,
        total_size=NR_NUM,
        observation=ST_NUM,
        amplitude=1,
        minor_amp=0.3,
    )

    block_noise_measurment = create_voxel_sampling(
        spatial_block_noise_response, neuron_recf_loc
    )

    block_noise_iem = IEM1D(channel_arr)
    block_noise_iem.fit(stimulus, block_noise_measurment)
    block_noisy_channel = block_noise_iem.decode(block_noise_measurment)
    diff = block_noisy_channel - channel_activation

    def get_three_metrics(X, Y):
        return (
            linear_CKA(X, Y),
            np.mean(global_neigbor_dice(X, Y)),
            global_distance_variance(X, Y),
        )

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

import pandas as pd

measurement_labels = [
    "recon/channel",
    "channel/neuron",
    "recon/neuron",
    "measure/channel",
    "measure/recon",
    "measure/neuron",
]

# CKA DataFrame
lcka_df = pd.DataFrame(
    list(
        zip(
            rc_lcka_list,
            nc_lcka_list,
            rn_lcka_list,
            mc_lcka_list,
            mr_lcka_list,
            mn_lcka_list,
        )
    ),
    columns=measurement_labels,
)

# Mean Neighbor Dice DataFrame
mnd_df = pd.DataFrame(
    list(
        zip(
            rc_mnd_list, nc_mnd_list, rn_mnd_list, mc_mnd_list, mr_mnd_list, mn_mnd_list
        )
    ),
    columns=measurement_labels,
)

# Mean Global Displacement DataFrame
mgd_df = pd.DataFrame(
    list(
        zip(
            rc_mgd_list, nc_mgd_list, rn_mgd_list, mc_mgd_list, mr_mgd_list, mn_mgd_list
        )
    ),
    columns=measurement_labels,
)

lcka_df.to_csv("result_lcka.csv")
mnd_df.to_csv("result_mnd.csv")
mgd_df.to_csv("result_mgd.csv")

import matplotlib.pyplot as plt

ax = plt.subplot()
for item in zip(
    rc_lcka_list, nc_lcka_list, rn_lcka_list, mc_lcka_list, mr_lcka_list, mn_lcka_list
):
    ax.plot(item, color="cyan", alpha=1)
plt.title("CKA")
plt.xticks(
    [0, 1, 2, 3, 4, 5],
    [
        "recon/channel",
        "channel/neuron",
        "recon/neuron",
        "measure/channel",
        "measure_recon",
        "measure/neuron",
    ],
)
plt.show()
ax = plt.subplot()
for item in zip(
    rc_mnd_list, nc_mnd_list, rn_mnd_list, mc_mnd_list, mr_mnd_list, mn_mnd_list
):
    ax.plot(item, alpha=0.3)
plt.title("Mean neighbor dice")
plt.xticks(
    [0, 1, 2, 3, 4, 5],
    [
        "recon/channel",
        "channel/neuron",
        "recon/neuron",
        "measure/channel",
        "measure_recon",
        "measure/neuron",
    ],
)
plt.show()
ax = plt.subplot()
for item in zip(
    rc_mgd_list, nc_mgd_list, rn_mgd_list, mc_mgd_list, mr_mgd_list, mn_mgd_list
):
    ax.plot(item, alpha=0.3)
plt.title("Mean global displacement")
plt.xticks(
    [0, 1, 2, 3, 4, 5],
    [
        "recon/channel",
        "channel/neuron",
        "recon/neuron",
        "measure/channel",
        "measure_recon",
        "measure/neuron",
    ],
)
plt.show()
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
