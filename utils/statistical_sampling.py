import numpy as np

MEASUREMENT_GRID_SIZE = 0.05


def create_voxel_sampling(
    neural_responses: np.ndarray,
    neuron_loc: np.ndarray,
    voxel_width: float = MEASUREMENT_GRID_SIZE,
    min_loc: float = -3,
    max_loc: float = 3,
    statistic: str = "mean",
):
    stats_dict = {
        "mean": np.mean,
        "median": np.median,
    }
    stat_func = stats_dict.get(statistic, np.mean)
    voxel_bounds = np.arange(min_loc, max_loc, voxel_width)
    all_channel = []
    for i in range(len(voxel_bounds)):
        # get the index for bounded location
        use_index = np.logical_and(
            neuron_loc >= voxel_bounds[i], neuron_loc <= voxel_bounds[i] + voxel_width
        )
        voxel_responses = stat_func(neural_responses[:, use_index], axis=1).reshape(
            -1, 1
        )
        all_channel.append(voxel_responses)
    measurement = np.hstack(all_channel)
    measurement[np.isnan(measurement)] = 0
    return measurement
