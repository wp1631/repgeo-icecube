import numpy as np
import scipy as sp
from scipy.stats import vonmises
from scipy.stats import norm
from functools import partial
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

STIM_MIN = -np.pi
STIM_MAX = np.pi
STIM_SMAPLE_NUM = 200
STIM_WRAP = True

CHANNEL_LOC_MIN = -np.pi
CHANNEL_LOC_MAX = np.pi
CHANNEL_NUM = 3
CHANNEL_WIDTH = 1

stim_space = np.linspace(STIM_MIN, STIM_MAX, STIM_SMAPLE_NUM)
channel_loc_space = np.linspace(
    CHANNEL_LOC_MIN, CHANNEL_LOC_MAX, CHANNEL_NUM + 1 if STIM_WRAP else CHANNEL_NUM
)
function_gen = norm if not STIM_WRAP else vonmises
arg_dict = {"scale": CHANNEL_WIDTH} if not STIM_WRAP else {"kappa": CHANNEL_WIDTH}
channels = [
    partial(function_gen.pdf, loc=channel_loc, **arg_dict)
    for channel_loc in channel_loc_space
]
channel_resps = list(map(lambda x: x(stim_space), channels))
resps_np = np.array(channel_resps).T
mds = MDS()

res = mds.fit_transform(resps_np)

plt.scatter(res[:, 0], res[:, 1])
plt.show()
