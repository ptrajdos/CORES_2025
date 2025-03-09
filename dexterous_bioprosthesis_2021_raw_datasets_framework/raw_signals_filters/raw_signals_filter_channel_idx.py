
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_filters.raw_signals_filter import RawSignalsFilter


class RawSignalsFilterChannelIdx(RawSignalsFilter):

    def __init__(self,indices_list) -> None:
        super().__init__()
        self.indices_list = indices_list
    
    def fit(self,raw_signals:RawSignals):
        """
        Does nothing.
        """
        pass

    def transform(self,raw_signals:RawSignals):
        filtered_signals = RawSignals()
        for raw_signal in raw_signals:
            np_signal = raw_signal.signal
            object_class = raw_signal.object_class
            timestamp = raw_signal.timestamp
            channel_names = raw_signal.channel_names

            np_signal = np_signal[:,self.indices_list]
            channel_names = [ channel_names[idx] for idx in self.indices_list]

            new_signal = RawSignal(signal= np_signal, object_class=object_class,
                                     channel_names= channel_names, timestamp=timestamp)
            filtered_signals.append(new_signal)

        return filtered_signals

        
