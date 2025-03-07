import pickle
import numpy as np
from scipy.linalg import lstsq
from scipy.stats import vonmises
from itertools import product

# Neural topographic space (real physical location of neurons)
TOPOG_MIN = 0
TOPOG_MAX = 100
TOPOG_NUM = 100
topographic_base_space = np.linspace(TOPOG_MIN, TOPOG_MAX, TOPOG_NUM)

# Neural tuning space
BASIS_MIN = 20
BASIS_MAX = 80
BASIS_NUM = 1000
basis_base_space = np.linspace(BASIS_MIN, BASIS_MAX, BASIS_NUM)

# Neuron parameters as the cartesian product of the two
NEURON_PARAMS = [
    item
    for item in product(
        topographic_base_space, topographic_base_space, basis_base_space
    )
]

# Hypothetical Channel defined on hypothetical stimulus space derived from stimulus
CHANNEL_NUM = 20
CHANNEL_MAX = 2 * np.pi
CHANNEL_MIN = 0
hypotethic_channel_loc = np.linspace(CHANNEL_MIN, CHANNEL_MAX, CHANNEL_NUM)

# Derived hypothetical stimulus properties
STIM_MIN = 0
STIM_MAX = 2 * np.pi
STIM_NUM = 1000
stimulus_base_space = np.linspace(STIM_MIN, STIM_MAX, STIM_NUM)

# Hypothetical Channel Response
CHANNEL_KAPPA = 3
hypothetic_stimulus_space = stimulus_base_space  # Here we use identity -> indicating case where hypothesis space form complete information of the system
hypothetic_channel_response = np.empty(shape=(STIM_NUM, CHANNEL_NUM))

for idx, loc in enumerate(hypotethic_channel_loc):
    hypothetic_channel_response[:, idx] = vonmises.pdf(
        hypothetic_stimulus_space, loc=loc, kappa=CHANNEL_KAPPA
    ).T

# Simulate fMRI data using Justin Gardner methods

## Neural responses come first
NEURON_KAPPA = 7
neural_response = np.empty(shape=(STIM_NUM, BASIS_NUM))
for idx, loc in enumerate(basis_base_space):
    neural_response[:, idx] = vonmises.pdf(
        hypothetic_stimulus_space, loc=loc, kappa=NEURON_KAPPA
    ).T
neural_response += np.random.normal(0, 1 / NEURON_KAPPA, neural_response.shape)

## The fMRI synthetic (plain wrong but serve purpose)
MEASUREMENT_DIM_LIST = np.logspace(0, 3, 10).astype(int)
DIM_TRIALS = 500

# Loop to get statistics
dim_list = []
trial_list = []
enc_err_list = []
enc_mat_list = []
ienc_mat_list = []
test_measurement_list = []
recovered_resp_list = []
base_neural_resp_list = []

for m_dim in MEASUREMENT_DIM_LIST:
    for i in range(DIM_TRIALS):
        print(m_dim, i)
        dim_list.append(m_dim)
        trial_list.append(i + 1)
        MEASUREMENT_DIM = m_dim
        MEASUREMENT_NOISE_SD = 0.2
        transformation_matrix = np.abs(
            np.random.normal(0, 1, size=(BASIS_NUM, MEASUREMENT_DIM))
        )
        measurement_signal = np.matmul(neural_response, transformation_matrix)
        measurement_signal += np.random.normal(
            0,
            MEASUREMENT_NOISE_SD / np.sqrt(MEASUREMENT_DIM),
            size=measurement_signal.shape,
        )

        ## IEM Recovery
        iem_encoding_mat = lstsq(hypothetic_channel_response, measurement_signal)[0]
        predicted_signal = np.matmul(hypothetic_channel_response, iem_encoding_mat)
        encoding_err = np.sqrt(np.mean((predicted_signal - measurement_signal) ** 2))
        enc_err_list.append(encoding_err)
        enc_mat_list.append(iem_encoding_mat)

        inverted_encoding_mat = np.linalg.pinv(iem_encoding_mat)
        ienc_mat_list.append(inverted_encoding_mat)

        base_neural_resp_list.append(neural_response)
        test_measurement_data = np.matmul(neural_response, transformation_matrix)
        test_measurement_data += np.random.normal(
            0,
            MEASUREMENT_NOISE_SD / np.sqrt(MEASUREMENT_DIM),
            size=test_measurement_data.shape,
        )
        test_measurement_list.append(test_measurement_data)

        recovered_channel_resp = np.matmul(test_measurement_data, inverted_encoding_mat)
        recovered_resp_list.append(recovered_channel_resp)

exp_data = {
    "dim": dim_list,
    "trial": trial_list,
    "enc_err": enc_err_list,
    "enc_mat": enc_mat_list,
    "ienc_mat": ienc_mat_list,
    "test_measure_data": test_measurement_list,
    "rec_resp_list": recovered_resp_list,
    "base_neural_resp": neural_response,
}

print("done")

with open("./temp/exp_data.pkl", "wb") as file:
    pickle.dump(exp_data, file)
