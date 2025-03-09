import abc
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signal import RawSignal

from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals
import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_interface import RawSignalsSpoilerInterface

class RawSignalsSpoiler(RawSignalsSpoilerInterface):

    def __init__(self, channels_spoiled_frac=0.1, snr=1) -> None:
        """
        Raw SignalsSpoiler -- introduces noises and other disturbances to signals

        Arguments:
        ---------
        channels_spoiled_frac -- fraction of channels to be contaminated or None
        snr - signal to noise ratio of the contaminations

        """
        super().__init__()
        self.channels_spoiled_frac = channels_spoiled_frac
        self.snr = snr
 
    def _random_channel_selection(self,raw_signal:RawSignal):
        """
        Return list of randomly selected channels

        Arguments:
        ----------
        raw_signals -- RawSignals object

        Returns:
        iterable of selected channel indices
        """
        if self.channels_spoiled_frac == 0:
            #Explicit zero is passed
            return []
        
        
        
        n_samples, n_channels = raw_signal.to_numpy().shape

        if self.channels_spoiled_frac is None:
            if n_channels >1:
                n_sel_channels = np.random.randint(1,n_channels)
            else:
                n_sel_channels = np.random.randint(n_channels)

            if n_sel_channels == 0:
                return []
        else:    
            n_sel_channels = int( np.min( [np.ceil(self.channels_spoiled_frac * n_channels), n_channels ]))

        sel_channel_indices = np.random.choice([*range(n_channels)], size=n_sel_channels, replace=False)

        return sel_channel_indices
    
    def _channel_powers(self, raw_signal_np):
        powers = np.mean( np.power(raw_signal_np,2) ,axis=0)
        return powers
    
    def _desired_channel_noise_powers(self,raw_signal_np):
        channel_powers = self._channel_powers(raw_signal_np)

        desired_powers = np.power( 10.0, np.log10(channel_powers) - self.snr/10.0 )
        
        return desired_powers
    
    def _calculate_snrs(self, raw_signal_np, noise_signal_np):
        signal_powers = np.mean( np.power(raw_signal_np,2) ,axis=0)
        noise_powers = np.mean( np.power(noise_signal_np,2) ,axis=0)

        snrs = 10.0* np.log10(signal_powers/noise_powers)

        return snrs






