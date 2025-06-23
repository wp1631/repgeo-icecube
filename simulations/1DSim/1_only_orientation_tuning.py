import numpy as np
from scipy.stats import vonmises
from scipy.linalg import lstsq
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from icecream import ic
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import seaborn as sns

# ============Define Parameters==============
NR_NUM = 3000

# Neuron Orientation Tuning
NR_OT_LOC_MIN = 0  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = 2 * np.pi
NR_OT_KAPPA = 10000

# Stimulus sample
ST_NUM = 1000
ST_OR_MIN = 0  # Stimulus orientation
ST_OR_MAX = 2 * np.pi

# Channel paraneters
CH_NUM = 6
CH_OR_LOC_MIN = 0
CH_OR_LOC_MAX = 2 * np.pi
CH_OR_KAPPA = 5


stimulus_ori = np.random.uniform(ST_OR_MIN, ST_OR_MAX, ST_NUM)
neuron_tuning_loc = np.random.uniform(NR_OT_LOC_MIN, NR_OT_LOC_MAX, NR_NUM)
neuron_tuning_kappa = np.full(NR_NUM, NR_OT_KAPPA)

ic(stimulus_ori.shape)
ic(neuron_tuning_loc.shape)
ic(neuron_tuning_kappa.shape)


def _get_response(orientation_tuning_loc: float, orientation_tuning_kappa: float):
    return vonmises.pdf(
        stimulus_ori, loc=orientation_tuning_loc, kappa=orientation_tuning_kappa
    )


neural_responses = [
    _get_response(x, y) for x, y in zip(neuron_tuning_loc, neuron_tuning_kappa)
]

neural_responses = np.array(neural_responses).T
ic(neural_responses.shape)
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

pca_embedding = PCA(n_components=3)
pca_embedded = pca_embedding.fit_transform(neural_responses_sorted.T)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    pca_embedded[:, 0],
    pca_embedded[:, 1],
    pca_embedded[:, 2],
    c=np.arange(NR_NUM),
    alpha=0.3,
)
plt.title("PCA Population Embedding of Sparse Orientation Coding, Neural Response (3D)")
plt.show()

population_embedding_mds = embedding.fit_transform(neural_responses_sorted.T)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    population_embedding_mds[:, 0],
    population_embedding_mds[:, 1],
    population_embedding_mds[:, 2],
    c=np.arange(NR_NUM),
    alpha=0.3,
)
plt.title("MDS Population Embedding of Sparse Orientation Coding, Neural Response (3D)")
plt.show()
neural_identity_sorted = neural_responses.T[np.argsort(neuron_tuning_loc)]
p_dist = pdist(neural_identity_sorted)
dist_mat = squareform(p_dist)
plt.imshow(dist_mat)
plt.colorbar()
plt.show()

plt.hist(neuron_tuning_loc, bins=20)
plt.show()
