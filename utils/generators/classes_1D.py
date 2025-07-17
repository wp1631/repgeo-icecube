import numpy as np
from scipy.stats import vonmises, norm
from dataclasses import dataclass
from icecream import ic


@dataclass
class Stimulus1D:
    stimulus_location: np.ndarray
    stimulus_orientation: np.ndarray
    stimulus_contrast: np.ndarray
    stimulus_size: np.ndarray


class NeuronArray1D:

    __slots__ = [
        "_neural_tuning_loc",
        "_neural_tuning_kappa",
        "_neural_tuning_amplitude",
        "_neural_recf_loc",
        "_neural_recf_width",
    ]

    _neural_tuning_loc: np.ndarray
    _neural_tuning_kappa: np.ndarray
    _neural_tuning_amplitude: np.ndarray

    _neural_recf_loc: np.ndarray
    _neural_recf_width: np.ndarray

    def __init__(
        self,
        tuning_loc: np.ndarray,
        tuning_kappa: np.ndarray,
        tuning_amplitude: np.ndarray,
        recf_loc: np.ndarray,
        recf_width: np.ndarray,
    ):
        self._neural_tuning_loc = tuning_loc
        self._neural_tuning_kappa = tuning_kappa
        self._neural_tuning_amplitude = tuning_amplitude
        self._neural_recf_loc = recf_loc
        self._neural_recf_width = recf_width

    def _get_res_amp(self, stimulus: Stimulus1D):
        return np.array(
            [
                norm.pdf(
                    stimulus.stimulus_location,
                    loc=loc,
                    scale=sc,
                )
                for loc, sc in zip(self._neural_recf_loc, self._neural_recf_width)
            ]
        ).T

    def get_response_amplitude(self, stimulus: Stimulus1D) -> np.ndarray:
        ic(self._get_res_amp(stimulus))
        return np.dot(
            self._get_res_amp(stimulus),
            self._neural_tuning_amplitude.reshape(-1, 1),
        )

    def _get_res_ori(self, stimulus: Stimulus1D):
        return np.array(
            [
                vonmises.pdf(
                    stimulus.stimulus_orientation,
                    loc=x,
                    kappa=y,
                )
                for x, y in zip(self._neural_tuning_loc, self._neural_tuning_kappa)
            ]
        ).T

    def get_response_orientation(self, stimulus: Stimulus1D) -> np.ndarray:
        return self._get_res_ori(stimulus)

    def get_two_responses(self, stimulus: Stimulus1D) -> tuple[np.ndarray, np.ndarray]:
        return self.get_response_amplitude(stimulus), self.get_response_orientation(
            stimulus
        )

    def get_responses(self, stimulus: Stimulus1D) -> np.ndarray:
        return (
            self.get_response_amplitude(stimulus)
            * self.get_response_orientation(stimulus)
        ).T

    def _get_res_deriv(self, stimulus: Stimulus1D):
        return np.array(
            [
                y
                * np.cos(stimulus.stimulus_orientation - x)
                * vonmises.pdf(stimulus.stimulus_orientation, loc=x, kappa=y)
                for x, y in zip(self._neural_tuning_loc, self._neural_tuning_kappa)
            ]
        ).T

    def get_derivatives(self, stimulus: Stimulus1D) -> np.ndarray:
        return self.get_response_amplitude(stimulus) * self._get_res_deriv(stimulus)
