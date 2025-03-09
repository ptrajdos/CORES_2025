import abc

from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals

class SetCreator(abc.ABC):
    """
    An interface for creating ordinary datasets from raw signal datasets
    """
    
    @abc.abstractmethod
    def fit_transform(self, raw_signals:RawSignals, y=None):
        """
        Create dataset from raw signals.
        Fits model if necesary and then transforms the raw_signals.

        Arguments:
        ---------

        raw_signals -- A RawSignals object
        y -- ignored

        Returns:
        -------

        A touple containig:
        X -- data
        y -- classes
        t -- timestamp
        """

    @abc.abstractmethod
    def fit(self, raw_signals:RawSignals, y=None):
        """
        Only fit the transformation model.

        Arguments:
        ---------

        raw_signals -- A RawSignals object
        y -- ignored

        Returns:
        SetCreator

        """

    @abc.abstractmethod
    def transform(self,raw_signals:RawSignals):
        """
        Create dataset from raw signals.
        Transform the raw_signals using previously fitted model.

        Arguments:
        ---------

        raw_signals -- A RawSignals object

        Returns:
        -------

        A touple containig:
        X -- data
        y -- classes
        t -- timestamp
        """

    @abc.abstractmethod
    def get_channel_attribs_indices(self):
        """
        Get indices of channel-specific attributes

        Returns:
        List containing lists of channel specific attributes. 
        Or None.
        None means that there is no simple mapping from channels to attributes in output set
        """