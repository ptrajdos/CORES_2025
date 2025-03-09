

from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler import RawSignalsSpoiler
from copy import deepcopy

class RawSignalsSpoilerDummy(RawSignalsSpoiler):

    def __init__(self, channels_spoiled_frac=0.1, snr=1) -> None:
        super().__init__(channels_spoiled_frac, snr)

    def fit(self,raw_signals:RawSignals):
        # Does nothing
        return self

    def transform(self, raw_signals:RawSignals):
        return deepcopy(raw_signals)