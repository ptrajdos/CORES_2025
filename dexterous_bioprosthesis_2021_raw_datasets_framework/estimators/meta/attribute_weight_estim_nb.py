import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.base import clone
from dexterous_bioprosthesis_2021_raw_datasets_framework.preprocessing.select_attributes_transformer import (
    SelectAttributesTransformer,
)
from kernelnb.estimators.estimatornb import EstimatorNB
from pt_outlier_probability.outlier_probability_estimator import (
    OutlierProbabilityEstimator,
)


class AttributeWeightEstimNB(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        model_prototype=None,
        outlier_detector_prototype=None,
        random_state=0,
        channel_features=None,
    ) -> None:
        """
        #Meta-classifier that uses attribute weights to perform classification.
        It uses base classifier and outlier detectors to estimate attribute weights.
        The method accepts transformed attributes not raw signals

        Arguments:

        ----------

        model_prototype -- base classifier prototype for the ensemble

        outlier_detector_prototype -- prototype for outlier detector

        random_state -- seed for the random generator

        channel_features -- a list that contains channel specific features

        """
        super().__init__()

        self.model_prototype = model_prototype
        self.outlier_detector_prototype = outlier_detector_prototype

        self.random_state = random_state
        self.channel_features = channel_features

    def _select_channel_group_features(self, channel_group_indices):
        """
        Selects features for channel group

        Attributes:
        -----------

        channel_group_indices -- iterable containing indices of channels constitutes the group.

        Returns:
        --------
        List containing combined features for each channel in the group.
        """

        all_selected_features = []

        for ch_idx in channel_group_indices:
            all_selected_features += self.channel_features[ch_idx]

        return all_selected_features

    def _preprocess_X_y(self, X, y):
        """
        If some class(es) has only one object, then append another copy of this object.
        This is to perform proper stratification in datasest with
        """
        uniq, counts = np.unique(y, return_counts=True)

        one_instance_labels = uniq[counts == 1]
        select_inndices = [True if i in one_instance_labels else False for i in y]

        Xc = np.append(X, X[select_inndices], axis=0)
        yc = np.append(y, y[select_inndices])

        return Xc, yc

    def _prepare_base_outlier_detectors(self, X, y):
        """
        Prepares base outlier detectors.
        One base outlier detector per channel.
        """
        n_channels = len(self.channel_features)

        effective_outlier_detector_prototype = (
            self.outlier_detector_prototype
            if self.outlier_detector_prototype is not None
            else IsolationForest()
        )
        self.outlier_detectors_ = np.zeros(n_channels, dtype=object)
        y_o = np.ones_like(y, dtype=np.int64)
        # create outlier detectors
        for channel_id in range(n_channels):
            column_indices = self._select_channel_group_features([channel_id])
            base_detector = Pipeline(
                [
                    (
                        "trans",
                        SelectAttributesTransformer(column_indices=column_indices),
                    ),
                    (
                        "classifier",
                        OutlierProbabilityEstimator(
                            clone(effective_outlier_detector_prototype)
                        ),
                    ),
                ]
            )
            base_detector.fit(X, y_o)
            self.outlier_detectors_[channel_id] = base_detector

    def _create_base_classifier(self, X, y):
        """
        Trains base classifier
        """

        self.base_classifier_ = (
            clone(self.model_prototype)
            if self.model_prototype is not None
            else EstimatorNB()
        )

        y_train = self.label_encoder_.transform(y)
        self.base_classifier_.fit(X, y_train)

    def fit(self, X, y=None):
        self.random_state_ = check_random_state(self.random_state)

        Xp, yp = self._preprocess_X_y(X, y)
        self.classes_ = np.unique(yp)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(yp)

        self._prepare_base_outlier_detectors(Xp, yp)

        self._create_base_classifier(Xp, yp)

        return self

    def predict(self, X, y=None):

        check_is_fitted(
            self,
            (
                "classes_",
                "base_classifier_",
                "outlier_detectors_",
                "random_state_",
                "label_encoder_",
            ),
        )

        soft_predictions = self.predict_proba(X)
        class_idxs = np.argmax(soft_predictions, axis=1)
        return self.label_encoder_.inverse_transform(class_idxs)

    def transform_weights(self, weights):
        """
        Transforms the weights if needed. 
        This version just copy them.
        Placeholder for use in subclasses.

        Arguments:
        ----------

        param:weights:np.ndarray -- weights (n_samples, n_attributes)


        Returns:
        --------

        transformed weights (n_samples, n_attributes)
        """
        return weights

    def predict_proba(self, X, y=None):

        check_is_fitted(
            self,
            (
                "classes_",
                "base_classifier_",
                "outlier_detectors_",
                "random_state_",
                "label_encoder_",
            ),
        )

        n_objects = X.shape[0]
        n_attributes = X.shape[1]
        n_channels = len(self.channel_features)

        attribute_weights = np.zeros((n_objects, n_attributes))

        for channel_id in range(n_channels):
            inlier_scores = self.outlier_detectors_[channel_id].predict_proba(X)[
                :, 1, np.newaxis
            ]
            column_indices = self._select_channel_group_features([channel_id])
            attribute_weights[:, column_indices] = inlier_scores

        n_classes = len(self.classes_)

        attribute_weights = self.transform_weights(attribute_weights)

        probas = np.zeros((n_objects, n_classes))
        for sample_idx in range(n_objects):
            probas[sample_idx, :] = self.base_classifier_._predict_proba(
                X[sample_idx : sample_idx + 1, :], attribute_weights[sample_idx, :]
            )

        return probas
