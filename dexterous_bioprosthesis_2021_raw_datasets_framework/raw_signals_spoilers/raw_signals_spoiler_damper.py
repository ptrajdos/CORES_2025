

from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler import RawSignalsSpoiler
from copy import deepcopy
import numpy as np

class RawSignalsSpoilerDamper(RawSignalsSpoiler):

    def __init__(self, channels_spoiled_frac=0.1, snr=1) -> None:
        super().__init__(channels_spoiled_frac, snr)

    def fit(self,raw_signals:RawSignals):
        # Does nothing
        return self

    def transform(self, raw_signals:RawSignals):
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:
            selected_channels_idxs = self._random_channel_selection(signal)

            channel_powers = self._channel_powers(signal.to_numpy())
            desired_channel_noise_powers = self._desired_channel_noise_powers(signal.to_numpy())

            alphas = np.sqrt(desired_channel_noise_powers/channel_powers) #channel power is also noise power

            np_sig = signal.to_numpy()
            np_sig[:,selected_channels_idxs] *= 1 - alphas[selected_channels_idxs]

        return copied_signals
        