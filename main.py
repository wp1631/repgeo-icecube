import seaborn as sns
import numpy as np
from scipy.stats import vonmises
from typing import Any, Literal
from functools import partial
import cv2
import inspect
import matplotlib.pyplot as plt


def generate_axis(start: int | float, stop: int | float, num: int = 100):
    return np.linspace(start, stop, num)


def repeat_axes(dim: int, start: int | float, stop: int | float, num: int = 100):
    return np.repeat(generate_axis(start, stop, num), dim)


def generate_neuron_descriptions(
    axes: np.ndarray,
    receptive_field_size: int | float,
    orientation_tuning: int | float,
    orientation_width: int | float,
):
    return ...


class Neuron:
    _loc: tuple
    _receptive_size: int | float
    _orientation_loc: int | float
    _oreintation_width: int | float


# Create an array from the
generated_axes = []
stimulus_vals = np.linspace(0, np.pi, 100)


def tuned_neuron(
    loc: int | float = 0, kappa: int | float = 1, units: Literal["deg", "rad"] = "rad"
) -> Any:
    if units == "deg":
        loc = loc * np.pi / 180
    return partial(vonmises.pdf, loc=loc, kappa=kappa)


# neurons = [tuned_neuron(loc) for loc in np.linspace(0, np.pi, 10)]
# activities = np.array([neuron(stimulus_vals) for neuron in neurons])
# print(activities)
