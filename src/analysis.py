import pickle
import numpy as np
import pandas as sp
from scipy.linalg import svdvals
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

with open("./temp/exp_data.pkl", "rb") as file:
    exp_data = pickle.load(file)

enc_svds = [svdvals(item) for item in exp_data["enc_mat"]]
ienc_svds = [svdvals(item) for item in exp_data["ienc_mat"]]
erra = exp_data["enc_err"]

enc_ranks = [np.log(np.sum(item > 1e-12) + 0.1) for item in enc_svds]
ienc_ranks = [np.log(np.sum(item > 1e-12) + 0.1) for item in ienc_svds]

test_measurements = exp_data["test_measure_data"]
predict_measurements = exp_data["predict_measure_data"]
rec_resps = exp_data["rec_resp"]

mesurement_svds = [svdvals(item) for item in test_measurements]
measurement_ranks = [np.log(np.sum(item > 1e-12) + 0.1) for item in mesurement_svds]
pred_measure_svds = [svdvals(item) for item in predict_measurements]
pred_measure_ranks = [np.log(np.sum(item > 1e-12) + 0.1) for item in pred_measure_svds]

rec_svds = [svdvals(item) for item in rec_resps]
rec_ranks = [np.log(np.sum(item > 1e-12) + 0.1) for item in rec_svds]
r2s = []

for idx, item in enumerate(test_measurements):
    c_mat = np.corrcoef(item, predict_measurements[idx]) ** 2
    r2s.append(c_mat[0, 0])

plot_df = pd.DataFrame(
    {
        "enc_rank": enc_ranks,
        "ienc_rank": ienc_ranks,
        "rec_rank": rec_ranks,
        "pred_rank": pred_measure_ranks,
        "measurement_rank": measurement_ranks,
        "enc_err": erra,
    }
)
# plot_df["D_rank_enc"] = plot_df["measurement_rank"] - plot_df["pred_rank"]

sns.pairplot(plot_df)


# sns.scatterplot(data = plot_df, x="enc_rank", y="ienc_rank")
plt.show()
