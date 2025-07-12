import numpy as np
from scipy.stats import vonmises
from scipy.linalg import lstsq
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from icecream import ic
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from utils.generators.classes_1D import NeuronArray1D, Stimulus1D

# ============Define Parameters==============
NR_NUM = 3000

# Neuron Orientation Tuning
NR_OT_LOC_MIN = 0  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = 2 * np.pi
NR_OT_KAPPA = 1
NR_LOC_W = 100

# Stimulus sample
ST_NUM = 1000
ST_OR_MIN = 0  # Stimulus orientation
ST_OR_MAX = 2 * np.pi
ST_LOC_MIN = -3
ST_LOC_MAX = 3

# Channel paraneters
CH_NUM = 6
CH_OR_LOC_MIN = 0
CH_OR_LOC_MAX = 2 * np.pi
CH_OR_KAPPA = 5

stimulus_ori = np.random.uniform(ST_OR_MIN, ST_OR_MAX, ST_NUM)
stimulus_loc = np.random.uniform(ST_LOC_MIN, ST_LOC_MAX, ST_NUM)
neuron_tuning_loc = np.random.uniform(NR_OT_LOC_MIN, NR_OT_LOC_MAX, NR_NUM)
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

neural_responses = neuron_arr.get_responses(stimulus)
neural_responses = np.array(neural_responses).T

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
