import numpy as np
from scipy.stats import vonmises
from itertools import product
from sklearn.manifold import MDS
from matplotlib import pyplot as plt 

TOPOG_MIN = 0
TOPOG_MAX = 100
TOPOG_NUM = 100
topographic_base_space = np.linspace(TOPOG_MIN, TOPOG_MAX, TOPOG_NUM)

BASIS_MIN = 20
BASIS_MAX = 80
BASIS_NUM = 100
basis_base_space = np.linspace(BASIS_MIN, BASIS_MAX, BASIS_NUM)

NEURON_PARAMS = [
    item
    for item in product(
        topographic_base_space, topographic_base_space, basis_base_space
    )
]


CHANNEL_NUM = 10
CHANNEL_MAX = 100
CHANNEL_MIN = 0
hypotethic_channel_loc = np.linspace(CHANNEL_MIN, CHANNEL_MAX, CHANNEL_NUM)

STIM_MIN = 0
STIM_MAX = 100
STIM_NUM = 1000
stimulus_base_space = np.linspace(STIM_MIN, STIM_MAX, STIM_NUM)

KAPPA = 3
hypothetic_stimulus_space = stimulus_base_space # Here we use identity -> indicating case where hypothesis space form complete information of the system
hypothetic_channel_response = np.empty(shape = (STIM_NUM, CHANNEL_NUM))

for idx,loc in enumerate(hypotethic_channel_loc):
    hypothetic_channel_response[:,idx] = vonmises.pdf(hypothetic_stimulus_space, loc=loc, kappa = KAPPA).T

embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(hypothetic_channel_response)

plt.scatter(X_transformed[:,0], X_transformed[:,1])
plt.show()
