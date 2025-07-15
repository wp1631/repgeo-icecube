from utils.generators.classes_1D import NeuronArray1D, Stimulus1D
import numpy as np
from scipy.linalg import lstsq


class IEM1D:
    _channel_arr: NeuronArray1D
    _enc_weight: np.ndarray
    _enc_residues: np.ndarray
    _dec_weight: np.ndarray

    def __init__(self, channel_arr: NeuronArray1D):
        self._channel_arr = channel_arr

    def fit(self, stimulus: Stimulus1D, measurement: np.ndarray) -> None:
        channel_activation = self._channel_arr.get_responses(stimulus=stimulus).T
        fit_res = lstsq(channel_activation, measurement)
        self._enc_weight = fit_res[0]
        self._enc_residues = fit_res[1]
        self._dec_weight = np.linalg.pinv(self._enc_weight)

    def decode(self, measurement: np.ndarray) -> np.ndarray:
        assert measurement.shape[1] == self._dec_weight.shape[0]
        return measurement @ self._dec_weight

    def encode(self, signal: np.ndarray) -> None:
        assert signal.shape[1] == self._enc_weight.shape[1]
        return signal @ self._enc_weight

    @property
    def channel_arr(self):
        return self.channel_arr

    @property
    def encode_weight(self):
        return self._enc_weight

    @property
    def decode_weight(self):
        return self._dec_weight

    @property
    def encode_residues(self):
        return self._enc_residues
