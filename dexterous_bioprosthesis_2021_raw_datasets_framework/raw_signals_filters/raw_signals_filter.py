import abc
import inspect
import warnings

from sklearn.base import defaultdict

from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import RawSignals

#TODO  res = transformer.fit_transform(X, y, **fit_params). Fitparams is needed!
#Compatibility with transformerMixin is needed!
#TODO remove it!
from sklearn.decomposition import PCA

class RawSignalsFilter(abc.ABC):
    """
    Class that represents an interface for filters
    """
    @abc.abstractmethod
    def fit(self, raw_signals: RawSignals,y = None, **fit_params)->None:
        """
        Fits the filter.
        Arguments:
        ---------
        raw_signals --- An object of RawSignals class to be fitted with
        """
        
    @abc.abstractclassmethod
    def transform(self,raw_signals: RawSignals)->RawSignals:
        """
        Transforms the RawSignals object
        Arguments:
        ---------
        raw_signals --- RawSignals object to be filtered

        Returns:
        --------
        RawSignals object
        """

    def fit_transform(self,raw_signals:RawSignals, y = None, **fit_params)->RawSignals:
        """
        Fits and then transforms the given RawSignals object

        Arguments:
        ---------
        raw_signals --- RawSignals object to be filtered

        Returns:
        --------
        RawSignals object
        """
        self.fit(raw_signals, **fit_params)
        return self.transform(raw_signals)
    
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            # TODO(1.4): remove specific handling of "base_estimator".
            # The "base_estimator" key is special. It was deprecated and
            # renamed to "estimator" for several estimators. This means we
            # need to translate it here and set sub-parameters on "estimator",
            # but only if the user did not explicitly set a value for
            # "base_estimator".
            if (
                key == "base_estimator"
                and valid_params[key] == "deprecated"
                and self.__module__.startswith("sklearn.")
            ):
                warnings.warn(
                    f"Parameter 'base_estimator' of {self.__class__.__name__} is"
                    " deprecated in favor of 'estimator'. See"
                    f" {self.__class__.__name__}'s docstring for more details.",
                    FutureWarning,
                    stacklevel=2,
                )
                key = "estimator"
            valid_params[key].set_params(**sub_params)

        return self