import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import svd
from scipy.stats import special_ortho_group

TOL = 1e-12

RANDOM_ROW = 10000
RANDOM_COL = 20

random_data = np.random.normal(loc=0, scale=1, size=(RANDOM_ROW, RANDOM_COL))

random_rot_mat = special_ortho_group(dim=RANDOM_COL).rvs()


def relu(x: int | float):
    if x > 0:
        return x
    else:
        return 0


vectorized_relu = np.vectorize(relu)

# rotated data should have the rank of min(row, col)
rotated_data = random_data @ random_rot_mat

# transformed data here should also have rank deficit (or not)
transformed_data = vectorized_relu(rotated_data)

# biased data here should have the same rank because we add the same vector all over
biased_transformed_data = vectorized_relu(
    rotated_data + np.random.normal(loc=0, scale=1, size=(1, RANDOM_COL))
)

# noisy data should have the maximum rank (the same as previous lol)
noised_data = biased_transformed_data + np.random.normal(
    loc=0, scale=1, size=biased_transformed_data.shape
)

L2_COL = 40
l2_transformation_matrix = np.random.normal(loc=0, scale=1, size=(RANDOM_COL, L2_COL))
l2_data = rotated_data @ l2_transformation_matrix
l2_noise_data = l2_data + np.random.normal(loc=0, scale=1, size=(RANDOM_ROW, L2_COL))
l2_relu = vectorized_relu(l2_noise_data)

pre_svd = svd(rotated_data)
pre_svdvalues = pre_svd[1]
pre_svdvectors = pre_svd[0]
pre_non_zeros_svdvals = np.sum(pre_svdvalues > TOL)

post_svd = svd(transformed_data)
post_svdvalues = post_svd[1]
post_svdvectors = post_svd[0]
post_non_zeros_svdvals = np.sum(post_svdvalues > TOL)

biased_transformed_svd = svd(biased_transformed_data)
biased_transformed_svdvals = biased_transformed_svd[1]
biased_transformed_svdvectors = biased_transformed_svd[0]
biased_transformed_non_zeros_svdvals = np.sum(biased_transformed_svdvals > TOL)

L2_data_svd = svd(l2_data)
l2_data_svdvalues = L2_data_svd[1]
l2_data_svdvectors = L2_data_svd[0]
l2_non_zeros_svdvals = np.sum(l2_data_svdvalues > TOL)

l2_noise_svd = svd(l2_noise_data)
l2_noise_svdvalues = l2_noise_svd[1]
l2_noise_svdvectors = l2_noise_svd[0]
l2_noise_non_zeros_svdvals = np.sum(l2_noise_svdvalues > TOL)

l2_relu_svd = svd(l2_relu)
l2_relu_svdvalues = l2_relu_svd[1]
l2_relu_svdvectors = l2_relu_svd[0]
l2_relu_non_zeros_svdvals = np.sum(l2_noise_svdvalues > TOL)

print(f"pre-transform: {pre_non_zeros_svdvals}")
print(f"post-transform: {post_non_zeros_svdvals}")
print(f"biased-transformed: {biased_transformed_non_zeros_svdvals}")
print(f"l2: {l2_non_zeros_svdvals}")
print(f"l2_noise: {l2_noise_non_zeros_svdvals}")
print(f"l2_relu: {l2_relu_non_zeros_svdvals}")
