from dexterous_bioprosthesis_2021_raw_datasets_framework.estimators.meta.attribute_weight_estim_nb import (
    AttributeWeightEstimNB,
)


class AttributeWeightEstimNBHard(AttributeWeightEstimNB):

    def __init__(
        self,
        model_prototype=None,
        outlier_detector_prototype=None,
        random_state=0,
        channel_features=None,
        threshold = 0.5
    ):
        """
        Meta-classifier that uses attribute weights to perform classification.
        It uses base classifier and outlier detectors to estimate attribute weights.
        The method accepts transformed attributes not raw signals

        Arguments:

        ----------

        model_prototype -- base classifier prototype for the ensemble

        outlier_detector_prototype -- prototype for outlier detector

        random_state -- seed for the random generator

        channel_features -- a list that contains channel specific features

        threshold -- decision threshold

        """
        super().__init__(
            model_prototype, outlier_detector_prototype, random_state, channel_features
        )

        self.threshold = threshold


    def transform_weights(self, weights):
        """
        Arguments:
        --------
        #param:weights:np.ndarray -- weights (n_samples, n_attributes)


        Returns:
        --------

        transformed weights (n_samples, n_attributes)
        """
        weights[weights < self.threshold] = 0
        weights[weights >= self.threshold] = 1
        return weights