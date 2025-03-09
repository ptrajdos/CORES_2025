from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class SelectAttributesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column_indices=[0]) -> None:
        """
        Selects a single attribute (column) from datases

        Arguments:
        column_number:int -- column to select

        """
        super().__init__()
        self.column_indices = column_indices


    def fit(self, X, y=None):
        """
        Does nothing
        """
        return self

    def transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            return X.iloc[:,self.column_indices]

        return X[:,self.column_indices]