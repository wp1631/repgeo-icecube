import numpy as np
import matplotlib.pyplot as plt
import pdb

from sklearn.utils.fixes import pd

observation_num = 10
observation_channel = 3
representational_dissimilarity_matrix = np.zeros((observation_num, observation_num))
to_fill_diag = np.full((observation_num), 1) + np.random.random(observation_num) * 0.001

np.fill_diagonal(representational_dissimilarity_matrix, to_fill_diag)
C = np.identity(observation_num) - (1 / observation_num) * np.full(
    observation_num, observation_num
)
B = -0.5 * (C.T @ representational_dissimilarity_matrix @ C)

dec = np.linalg.eig(B)
breakpoint()
ein_values, ein_vectors = dec.eigenvalues, dec.eigenvectors
print(ein_values)
breakpoint()
ld = np.zeros((observation_num, observation_num))
np.fill_diagonal(ld, ein_values)
breakpoint()
X = ein_vectors @ np.sqrt(ld)
breakpoint()
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])
plt.show()
