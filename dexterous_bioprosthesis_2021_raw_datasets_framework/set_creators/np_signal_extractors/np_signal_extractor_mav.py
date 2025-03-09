
from dexterous_bioprosthesis_2021_raw_datasets_framework.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np


class NpSignalExtractorMav(NPSignalExtractor):
    
    def fit(self, X):
        return self
    
    def transform(self, X):

        return np.mean(np.abs(X), axis=0)

    def attribs_per_column(self):
        return 1