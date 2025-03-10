from utils.block_matrix import generate_block
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from sklearn.manifold import MDS

MAX_BLOCK = 10
ROW_NUM = 20
COL_NUM = 5
MODE = "row"
mat_block = []

for i in range(MAX_BLOCK):
    l_block_num = i
    r_block_num = (MAX_BLOCK - i) - 1
    c_block = generate_block(size=(ROW_NUM, COL_NUM), mode=MODE)
    l = [np.zeros((ROW_NUM, COL_NUM))] * l_block_num
    r = [np.zeros((ROW_NUM, COL_NUM))] * r_block_num
    row = [*l, c_block, *r]
    mat_block.append(row)

mat = np.block(mat_block)
svd_data = svd(mat)
svd_vec = svd_data[0]

svd_vec /= np.sum(svd_vec, axis=1)
emb = MDS(n_components=3)
svd_vec_emb = emb.fit_transform(svd_vec)
svd_vals = svd_data[1]

plt.scatter(
    svd_vec_emb[: svd_vals.shape[0], 0],
    svd_vec_emb[: svd_vals.shape[0], 1],
    c=svd_vals[: svd_vals.shape[0]],
)
plt.colorbar()
plt.show()
