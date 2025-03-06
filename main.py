import numpy as np
from scipy.stats import vonmises
from itertools import product
from sklearn.manifold import MDS
from matplotlib import pyplot as plt

# Neural topographic space (real physical location of neurons)
TOPOG_MIN = 0
TOPOG_MAX = 100
TOPOG_NUM = 100
topographic_base_space = np.linspace(TOPOG_MIN, TOPOG_MAX, TOPOG_NUM)

# Neural tuning space
BASIS_MIN = 20
BASIS_MAX = 80
BASIS_NUM = 100
basis_base_space = np.linspace(BASIS_MIN, BASIS_MAX, BASIS_NUM)

# Neuron parameters as the cartesian product of the two
NEURON_PARAMS = [
    item
    for item in product(
        topographic_base_space, topographic_base_space, basis_base_space
    )
]

# Hypothetical Channel defined on hypothetical stimulus space derived from stimulus
CHANNEL_NUM = 100
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
NEURON_KAPPA = 7
neural_response = np.empty(shape=(STIM_NUM, BASIS_NUM))
for idx, loc in enumerate(basis_base_space):
    neural_response[:, idx] = vonmises.pdf(
        hypothetic_stimulus_space, loc=loc, kappa=NEURON_KAPPA
    ).T
neural_response += np.random.normal(0, 1 / NEURON_KAPPA, neural_response.shape)

# Visualization
embedding = MDS(n_components=2)
channel_resp_mds = embedding.fit_transform(hypothetic_channel_response)
neural_resp_mds = embedding.fit_transform(neural_response)

fig, ax = plt.subplots(figsize=(16, 6), ncols=2)
ax[0].scatter(neural_resp_mds[:, 0], neural_resp_mds[:, 1])
ax[1].scatter(channel_resp_mds[:, 0], channel_resp_mds[:, 1])
plt.show()
