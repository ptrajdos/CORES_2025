

from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler import RawSignalsSpoiler
from copy import deepcopy
import numpy as np

class RawSignalsSpoilerSine(RawSignalsSpoiler):

    def __init__(self, channels_spoiled_frac=0.1, snr=1, frequency=50, freq_deviation=2) -> None:
        super().__init__(channels_spoiled_frac, snr)
        self.frequency = frequency
        self.frequency_deviation = freq_deviation

    def fit(self,raw_signals:RawSignals):
        # Does nothing
        return self

    def transform(self, raw_signals:RawSignals):
        copied_signals = deepcopy(raw_signals)

        used_frequency = 2 * self.frequency_deviation * np.random.random(1) + self.frequency - self.frequency_deviation
        phase_shift = 4 * np.pi * np.random.random(1) - 2* np.pi
        
        for signal in copied_signals:
            selected_channels_idxs = self._random_channel_selection(signal)
            n_samples, n_channels = signal.to_numpy().shape

            x = np.linspace(0,n_samples, n_samples) * 1/raw_signals.sample_rate
            noise_sig_np_col = np.reshape(np.sin( 2* np.pi * used_frequency * x + phase_shift), (n_samples,-1))
            noise_sig_np = np.repeat(noise_sig_np_col, n_channels, axis=1)
            acc_noise_powers = self._channel_powers(noise_sig_np)

            np_sig = signal.to_numpy()
            desired_noise_powers = self._desired_channel_noise_powers(np_sig)

            alphas = np.sqrt(desired_noise_powers/acc_noise_powers)

            np_sig[:,selected_channels_idxs] += alphas[selected_channels_idxs] * noise_sig_np[:,selected_channels_idxs]

        return copied_signals
        