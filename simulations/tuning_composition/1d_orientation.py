from scipy.stats import vonmises
import jax.numpy as jnp
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np

channel_kappa = 2
neuron_kappa = 10
sample_neuron = 1000

empirical_tuning_location = vonmises.rvs(
    loc=jnp.pi, kappa=channel_kappa, size=sample_neuron
)
probe_num = 3000
probe_stimulus_value = jnp.linspace(0, 2 * jnp.pi, probe_num)

noise_amp = 0.05

responses = []
for idx, val in enumerate(empirical_tuning_location):
    responses.append(vonmises.pdf(probe_stimulus_value, loc=val, kappa=neuron_kappa))
responses_arr = jnp.array(responses).T + noise_amp * np.random.random(
    size=(probe_num, sample_neuron)
)

mean_responses = responses_arr.mean(axis=1)

plt.hist(jnp.pi + empirical_tuning_location)
plt.show()

fig, axes = plt.subplots(ncols=2, sharey=True)
axes[0].scatter(probe_stimulus_value, mean_responses, alpha=0.05)
axes[0].set_title("Average population responses")
axes[1].scatter(
    probe_stimulus_value,
    vonmises.pdf(probe_stimulus_value, loc=jnp.pi, kappa=channel_kappa)
    + noise_amp * np.random.random(size=(probe_num)),
    alpha=0.05,
)
axes[1].set_title("Channel responses over stimulus")
fig.suptitle(
    "Average population signal strength Vs Channel responses  over stimulus orientation"
)
plt.show()


embedding = MDS(n_components=3)
population_embedding_mds = embedding.fit_transform(responses_arr)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    population_embedding_mds[:, 0],
    population_embedding_mds[:, 1],
    population_embedding_mds[:, 2],
    c=jnp.arange(probe_num),
    alpha=0.3,
    cmap="hsv",
)
plt.title("MDS Population Embedding of Neural Response (3D)")
plt.show()
