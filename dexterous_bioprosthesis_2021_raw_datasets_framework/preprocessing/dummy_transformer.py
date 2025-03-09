from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy

class DummyTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """
        Does nothing
        """
        return self

    def transform(self, X, y=None):
        return deepcopy(X)
    
    def fit_transform(self, X, y=None, **fit_params):
        return deepcopy(X)
