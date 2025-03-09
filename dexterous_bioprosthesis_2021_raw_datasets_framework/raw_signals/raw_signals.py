
from copy import deepcopy
import numpy as np
from .raw_signal import RawSignal
from collections.abc import Iterable
from collections.abc import Collection
class RawSignals():
    """
     Class represents a dataset of raw signals
    """

    def __init__(self, raw_signal_list= None, sample_rate=1000) -> None:
        """
        Creates a new instance of the class
        """
        self.raw_signals_list = list()
        self.sample_rate = sample_rate

        if raw_signal_list is not None:
            for sig in raw_signal_list:
                sig.set_sample_rate(self.sample_rate)
                self.append(sig)

    def __iter__(self):
        return iter(self.raw_signals_list)

    def __getitem__(self,key):

        if not isinstance(key, tuple):
            if isinstance(key, slice):
                return RawSignals(self.raw_signals_list[key], sample_rate=self.sample_rate)

            if isinstance(key, Collection):
                if len(key) == len(self.raw_signals_list) and all( isinstance(item, bool) or isinstance(item, np.bool_) for item in key):
                    sel_list = [self.raw_signals_list[i_idx] for i_idx,i in enumerate( key ) if i ]
                else:
                    sel_list = [ self.raw_signals_list[i] for i in key ]
                    
                return RawSignals(sel_list,sample_rate=self.sample_rate)

            if isinstance(key,Iterable):
                sel_list = [ self.raw_signals_list[i] for i in key ]
                return RawSignals(sel_list,sample_rate=self.sample_rate)

            return self.raw_signals_list[key]
        
        # Is tuple here

        if len(key) == 1:
            return self.__getitem__(key[0])
    
        selected_signals = self.__getitem__(key[0])

        if isinstance(selected_signals, RawSignal):
            return selected_signals[key[1:]]
        
        return RawSignals( sample_rate=self.sample_rate, raw_signal_list= [i[key[1:]] for i in selected_signals] )


        


    def append(self, other:RawSignal):
        """
        Appends another RawSignal object to the dataset

        Arguments:
        ----------
        other -- object to add

        Raises:
        ------
        Value error if other object is not a RawSignal instance

        """
        if not isinstance(other, RawSignal):
            raise ValueError("Object is not a raw signal")

        if not hasattr(self,'signal_n_cols'):
            self.signal_n_cols = other.signal.shape[1]

        if self.signal_n_cols != other.signal.shape[1]:
            raise ValueError("Appending signal with diferent number of columns")

        #TODO Why copy? It is inconsistent with __iadd__ method
        other_copy = deepcopy(other)
        other_copy.set_sample_rate(self.sample_rate)
        self.raw_signals_list.append(other_copy)

    def __iadd__(self,other):
        """
        Operator for += on an another RawSignals object
        """
        for it in other:
            self.append(it)

        return self

    def __len__(self):
        return len(self.raw_signals_list)

    def __eq__(self, __o: object) -> bool:

        if id(self) == id(__o):
            return True

        if type(self) != type(__o):
            return False

        
        return self.raw_signals_list == __o.raw_signals_list

    def get_labels(self):
        """
        Returns:
        -------

        List of labels of stored signals
        """

        return np.asanyarray([ rs.get_label() for rs in self.raw_signals_list])

    def get_timestamps(self):
        """
        Returns:
        --------

        List of timestamps of the stored signals.

        """
        return np.asanyarray([ rs.get_timestamp() for rs in self.raw_signals_list])
    
    def set_labels(self, labels):
        """
        Set labels for all raw_signals.
        
        Arguments:
        labels -- iterable with new labels. Must contain exact the same number of values as the number of 
            objects in the set.
        """
        if len(labels) != len(self):
            raise ValueError("Wrong number of labels.")
        
        for sig, label in zip(iter(self), labels):
            sig.set_label(label)

    def initialize_empty(self):
        """
        Initializes new, empty RawSignals object

        Returns:
        RawSignals with empty signals list
        """
        return RawSignals(raw_signal_list=[], sample_rate=self.sample_rate)
    def get_sample_rate(self):
        return self.sample_rate
    
    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

        for sig  in iter(self):
            sig.set_sample_rate(self.sample_rate)

    def to_numpy(self):
        """
        Returns the raw_signals object as a numpy array.
        Assuming that all raw_signal object have the same dimension

        Returns:
        ---------
        numpy array of shape (n_objects, n_samples, n_channels)

        """
        np_array = np.asanyarray([ rs.to_numpy() for rs in self.raw_signals_list])
        return np_array