from utils.block_matrix import generate_block
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.imshow(mat)
plt.show()
