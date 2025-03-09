import abc
import numpy as np
from sklearn.utils.validation import check_is_fitted

class OutlierGenerator(abc.ABC):

    def __init__(self, outlier_label=-1) -> None:
        super().__init__()
        self.outlier_label_prototype = outlier_label

    @abc.abstractmethod
    def fit(self,X,y):
        """
        Fits Outlier generator

        Returns:
        self
        """
        np_labels = np.asanyarray(y)
        self.outlier_label_dtype_ = np_labels.dtype
        self.outlier_label_ = np.asanyarray([self.outlier_label_prototype]).astype(self.outlier_label_dtype_)
        

        return self



    @abc.abstractmethod
    def generate(self):
        """
        Generates outliers
        Returns:
        tuple X,y 
        """
        check_is_fitted(self, ("outlier_label_","outlier_label_dtype_"))
        return None

    def fit_generate(self,X,y):
        return self.fit(X,y).generate()