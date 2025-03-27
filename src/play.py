import numpy as np

spins = np.empty(shape=(10, 10))
random_index = tuple(np.random.randint(dim_size + 1) for dim_size in spins.shape)
print(random_index)
