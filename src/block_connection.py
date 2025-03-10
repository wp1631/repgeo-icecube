from utils.block_matrix import generate_block
import numpy as np
from sklearn.manifold import MDS
from plotly.express import scatter_3d

MAX_BLOCK = 5
ROW_NUM = 200
COL_NUM = 20
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
svd_vec = svd_data[0]  # These vectors are already normalized
emb = MDS(n_components=3)
svd_vals = svd_data[1] / np.sum(svd_data[1])
svd_vec_emb = emb.fit_transform(svd_vec)[: svd_vals.shape[0]]

use_svd_vals = svd_vals[: np.min(svd_vals.shape)]

fig = scatter_3d(
    x=svd_vec_emb[:, 0],
    y=svd_vec_emb[:, 1],
    z=svd_vec_emb[:, 2],
    color=use_svd_vals,
)
fig.show()
