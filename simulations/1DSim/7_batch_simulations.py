from typing import Tuple
import numpy as np
from itertools import product
from utils.generators.classes_1D import NeuronArray1D, Stimulus1D
from utils.generators.noise import create_block_noise
from utils.statistical_sampling import create_voxel_sampling
from utils.iem import IEM1D
from utils.rep_metrics import global_distance_variance, global_neigbor_dice, linear_CKA

# Neuron Orientation Tuning
NR_OT_LOC_MIN = -np.pi  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = np.pi

# Stimulus sample
ST_NUM = 1000
ST_OR_MIN = -np.pi  # stimulus.stimulus_orientation
ST_OR_MAX = np.pi
ST_LOC_MIN = -3
ST_LOC_MAX = 3

# Channel paraneters
CH_OR_LOC_MIN = -np.pi
CH_OR_LOC_MAX = np.pi
CH_RECF_MIN = -3
CH_RECF_MAX = 3

# Measurement
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
SP_MEASUREMENT_NOISE_AMP = np.logspace(-2, 2, endpoint=True, base=10, num=5)


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

    all_res = []

    # channel vs recon (cr)
    _ = get_three_metrics(channel_activation, block_noisy_channel)
    all_res.append(_[0])
    all_res.append(_[1])
    all_res.append(_[2])

    # channel vs neuron (cn)
    _ = get_three_metrics(channel_activation, spatial_block_noise_response)
    all_res.append(_[0])
    all_res.append(_[1])
    all_res.append(_[2])

    # recon vs neuron (rn)
    _ = get_three_metrics(block_noisy_channel, spatial_block_noise_response)
    all_res.append(_[0])
    all_res.append(_[1])
    all_res.append(_[2])

    # measure vs channel (mc)
    _ = get_three_metrics(block_noise_measurment, channel_activation)
    all_res.append(_[0])
    all_res.append(_[1])
    all_res.append(_[2])

    # measure vs recon (mr)
    _ = get_three_metrics(block_noise_measurment, block_noisy_channel)
    all_res.append(_[0])
    all_res.append(_[1])
    all_res.append(_[2])

    # measurement vs neuron (mn)
    _ = get_three_metrics(block_noise_measurment, spatial_block_noise_response)
    all_res.append(_[0])
    all_res.append(_[1])
    all_res.append(_[2])

    return all_res


from io import TextIOWrapper


def write_values(file: TextIOWrapper, *args, auto_close: bool = False):
    file.write(f"{','.join(map(str, args))}")
    file.write("\n")
    if auto_close:
        file.close()


if __name__ == "__main__":
    param_file = open("param_file.txt", "w")
    result_file = open("metrics_file.txt", "w")
    err_file = open("err_file.txt", "w")
    for (
        NR_NUM,
        NR_OT_KAPPA,
        NR_RECF_W,
        CH_NUM,
        CH_OT_KAPPA,
        CH_RECF_W,
        MEASUREMENT_GRID_SIZE,
        BLOCK_NOISE_AMP,
        BLOCK_NEURON_SIZE,
        BLOCK_NOISE_MINOR_AMP,
        MEASURE_NOISE_AMP,
        REP,
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
        SP_MEASUREMENT_NOISE_AMP,
        np.arange(REPEAT_NUM) + 1,
    ):
        try:
            # write parameters_file
            _res = generate_metrics(
                NR_NUM,
                NR_OT_KAPPA,
                NR_RECF_W,
                1000,  # number of stim
                CH_NUM,
                CH_OT_KAPPA,
                CH_RECF_W,
                CH_RECF_MIN,
                CH_RECF_MAX,
                MEASUREMENT_GRID_SIZE,
                BLOCK_NOISE_AMP,
                BLOCK_NEURON_SIZE,
                BLOCK_NOISE_MINOR_AMP,
                MEASURE_NOISE_AMP,
            )
            write_values(result_file, *_res)
            write_values(
                param_file,
                NR_NUM,
                NR_OT_KAPPA,
                NR_RECF_W,
                CH_NUM,
                CH_OT_KAPPA,
                CH_RECF_W,
                MEASUREMENT_GRID_SIZE,
                BLOCK_NOISE_AMP,
                BLOCK_NEURON_SIZE,
                BLOCK_NOISE_MINOR_AMP,
                MEASURE_NOISE_AMP,
                REP,
            )
        except:
            write_values(
                err_file,
                NR_NUM,
                NR_OT_KAPPA,
                NR_RECF_W,
                1000,  # number of stim
                CH_NUM,
                CH_OT_KAPPA,
                CH_RECF_W,
                CH_RECF_MIN,
                CH_RECF_MAX,
                MEASUREMENT_GRID_SIZE,
                BLOCK_NOISE_AMP,
                BLOCK_NEURON_SIZE,
                BLOCK_NOISE_MINOR_AMP,
                MEASURE_NOISE_AMP,
            )
    err_file.close()
    param_file.close()
    result_file.close()
