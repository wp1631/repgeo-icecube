import numpy as np
from scipy.linalg import lstsq
from scipy.stats import special_ortho_group, vonmises
from icecream import ic
from utils.generators.classes_1D import NeuronArray1D, Stimulus1D
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


# ============Visualization==============
# plot neural tuning profile
plot_neural_orientation_tuning_profile(
    stimulus, neuron_tuning_loc, neuron_tuning_kappa, neuron_tuning_amp, plot_every=100
)

# get neural responses
neural_responses = neuron_arr.get_responses(stimulus)
neural_responses = np.array(neural_responses).T

deriv = neuron_arr.get_derivatives(stimulus)
deriv_normed = np.linalg.norm(deriv, axis=1)
fisher_info = deriv_normed / np.sqrt(NR_NUM)

plot_orientation_activation(neural_responses, np.argsort(stimulus.orientation))

plot_orientation_fisher_information(stimulus, fisher_info)

# Plot MDS of the neural responses
plot_mds(neural_responses, c=stimulus.orientation)

# Plot MDS of the orientation gradient
plot_mds(deriv, c=stimulus.orientation, title="MDS Embedding of the Gradient (3D)")

# Plot MDS of the location gradient
sort_index = np.argsort(stimulus_ori)

plot_RDM(neural_responses, sort_index)

plot_representational_distance(stimulus.orientation, neural_responses)

plot_pca_scree(neural_responses)


# ============Measurement Simulation===============

MEASUREMENT_GRID_SIZE = 0.05

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

plot_mds(reconstructed_channal_responses, title="Reconstructed Channel Responses")

plot_representational_distance(stimulus.orientation, reconstructed_channal_responses)

# ================Covariate Noise==================
