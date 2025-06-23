import numpy as np
from numpy.ma import shape
from scipy.spatial import distance_matrix
from scipy.stats import vonmises
from scipy.linalg import lstsq
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from multiprocessing import Pool
from icecream import ic
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# ============Define Parameters==============
NR_NUM = 10000

# Neuron Orientation Tuning
NR_OT_LOC_MIN = 0  # Neuron minimum orientation tuning location
NR_OT_LOC_MAX = np.pi
NR_OT_KAPPA = 10

# Stimulus sample
ST_NUM = 1000
ST_OR_MIN = 0  # Stimulus orientation
ST_OR_MAX = np.pi

# Channel paraneters
CH_NUM = 6
CH_OR_LOC_MIN = 0
CH_OR_LOC_MAX = np.pi
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


if __name__ == "__main__":
    with Pool() as p:
        neural_responses = p.starmap(
            _get_response, zip(neuron_tuning_loc, neuron_tuning_kappa)
        )
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
    )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    plt.title("MDS Embedding of Sparse Orientation Coding, Neural Response MDS (3D)")
    plt.show()

    p_dist = pdist(neural_responses)
    dist_mat = squareform(p_dist)
    plt.imshow(dist_mat)
    plt.show()
