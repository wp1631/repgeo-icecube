import numpy as np
import scipy as sp
from scipy.stats import vonmises
from scipy.stats import norm
from functools import partial
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

STIM_MIN = -np.pi / 2
STIM_MAX = np.pi / 2
STIM_SMAPLE_NUM = 500
STIM_WRAP = True

CHANNEL_LOC_MIN = -np.pi
CHANNEL_LOC_MAX = np.pi
CHANNEL_NUM = 2
KAPPA_NUM = 7
KAPPA_SPACE = [1e-0]
# KAPPA_SPACE = np.logspace(-3,0,KAPPA_NUM)
all_resps = []
use_cmap = "hsv" if STIM_WRAP else "jet"
for CHANNEL_WIDTH in KAPPA_SPACE:
    stim_space = np.linspace(
        STIM_MIN, STIM_MAX, STIM_SMAPLE_NUM, endpoint=True if not STIM_WRAP else False
    )
    # channel_loc_space = np.linspace(CHANNEL_LOC_MIN, CHANNEL_LOC_MAX, CHANNEL_NUM + 1 if STIM_WRAP else CHANNEL_NUM)
    channel_loc_space = [-np.pi, -np.pi / 2]
    function_gen = norm if not STIM_WRAP else vonmises
    arg_dict = {"scale": CHANNEL_WIDTH} if not STIM_WRAP else {"kappa": CHANNEL_WIDTH}
    channels = [
        partial(function_gen.pdf, loc=channel_loc, **arg_dict)
        for channel_loc in channel_loc_space
    ]
    channel_resps = list(map(lambda x: x(2 * stim_space), channels))
    resps_np = np.array(channel_resps).T
    all_resps.append(resps_np)
mds = MDS()
agg_dat = np.vstack(all_resps)
res = mds.fit_transform(agg_dat)
fig, ax = plt.subplots()
for i in range(len(KAPPA_SPACE)):
    ax.scatter(
        res[i * STIM_SMAPLE_NUM : (i + 1) * STIM_SMAPLE_NUM, 0],
        res[i * STIM_SMAPLE_NUM : (i + 1) * STIM_SMAPLE_NUM, 1],
        label=f"k={KAPPA_SPACE[i]}",
        c=stim_space,
        s=2 + 5 * (KAPPA_NUM - i),
        alpha=(i + 1) / (KAPPA_NUM + 1),
        cmap=use_cmap,
    )
ax.set_title("MDS recovered Geometry of 2 von-mises encoder with kappa {kappa}")
plt.show()
