import abc


class NPSignalExtractor(abc.ABC):

    @abc.abstractmethod
    def fit(self, X):
        pass

    @abc.abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self,X):
        return self.fit(X).transform(X)

    @abc.abstractmethod
    def attribs_per_column(self):
        pass