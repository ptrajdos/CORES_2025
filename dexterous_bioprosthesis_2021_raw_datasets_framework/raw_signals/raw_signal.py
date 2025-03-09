import numpy as np
from copy import deepcopy

from collections.abc import Iterable
from collections.abc import Collection

class RawSignal():
    """
    Class represents raw signal (in the time domain).
    The signal may contain single or multiple channels.
    """
    def __init__(self, signal, object_class=np.nan,channel_names=None, timestamp=0, sample_rate = 1000):
        """
         Initializaes the class instance
         Arguments:
         ----------
         signal -- numpy array samples (rows) x channels (columns)
         object_class -- int/string represents class
         timestamp -- unix timestamp or zero
         sample_rate -- sample rate of the signal
        """

        self.signal = signal
        self.object_class = object_class
        self.timestamp = timestamp
        self.sample_rate = sample_rate

        if channel_names is None:
            self.channel_names = ["C{}".format(i) for i in range(signal.shape[1])]
        else:
            self.channel_names = channel_names

    def __eq__(self, __o: object) -> bool:

        if id(self) == id(__o):
            return True

        if  type(self) != type(__o):
            return False

        if not np.allclose(self.signal, __o.signal, rtol=1E-6, atol=1E-6):
            return False

        if self.object_class != __o.object_class:
            return False

        if self.timestamp != __o.timestamp:
            return False

        if not self.channel_names == __o.channel_names:
            return False
        
        if self.sample_rate != __o.sample_rate:
            return False

        return True
        
    def __getitem__(self, pos):
        """
        Returns a sliced object

        Arguments:
        pos -- slice, number of tuple of those

        """

        if isinstance(pos, tuple):
            if len(pos) == 1:
                return RawSignal(signal= self.signal[pos],
                         object_class= deepcopy(self.object_class), 
                         channel_names=  deepcopy(self.channel_names),
                         timestamp= deepcopy(self.timestamp),
                         sample_rate= deepcopy(self.sample_rate), 
                         )

            if len(pos) > 2:
                raise IndexError("Too many indices for the object. Object support 2d indexing, {} indices given".format(len(pos)))
        
            row_idx, col_idx = pos

            return RawSignal(signal= self.signal[pos],
                         object_class= deepcopy(self.object_class), 
                         channel_names=  self._column_names_get(col_idx),
                         timestamp= deepcopy(self.timestamp),
                         sample_rate= deepcopy(self.sample_rate), 
                         )
    

        return RawSignal(signal= self.signal[pos],
                         object_class= deepcopy(self.object_class), 
                         channel_names=  deepcopy(self.channel_names),
                         timestamp= deepcopy(self.timestamp),
                         sample_rate= deepcopy(self.sample_rate), 
                         )
    

    def _column_names_get(self, key):
        if isinstance(key, slice):
            return self.channel_names[key]

        if isinstance(key, Collection):
            if len(key) == len(self.channel_names) and all( isinstance(item, bool) or isinstance(item, np.bool_) for item in key):
                sel_list = [self.channel_names[i_idx] for i_idx,i in enumerate( key ) if i ]
            else:
                sel_list = [ self.channel_names[i] for i in key ]
                
            return sel_list


        return self.channel_names[key]
    
    def __len__(self):
        return self.signal.shape[0]
    
    def to_numpy(self):
        """
        Returns: signal representation as an numpy ndarray
        """
        return self.signal
    
    def get_label(self):
        return self.object_class
    
    def get_timestamp(self):
        return self.timestamp
    
    def set_label(self,label):
        self.object_class = label

    def get_sample_rate(self):
        return self.sample_rate
    
    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate