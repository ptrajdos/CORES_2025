from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals.raw_signals import (
    RawSignals,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_dummy import (
    RawSignalsSpoilerDummy,
)
from dexterous_bioprosthesis_2021_raw_datasets_framework.raw_signals_spoilers.raw_signals_spoiler_interface import (
    RawSignalsSpoilerInterface,
)

import numpy as np
from copy import deepcopy


class RawSignalsSpoilerMultiple(RawSignalsSpoilerInterface):

    def __init__(
        self,
        spoilers=[RawSignalsSpoilerDummy()],
        spoilers_weights=None,
        spoiled_fraction: float = 1.0,
        spoiler_relabalers=None,
    ) -> None:
        """
        Apply multiple signal spoilers.
        For each of the selected samples, the algorithm selects one label spoiler and apply it to the sample.

        Arguments:
        ----------

        spoilers -- iterable of spoiler objects to apply
        spoiler_weights -- iterable of apoiler weights or None for uniform weights
        spoiled_fraction -- how many samples from original set will be used to generate spoiled signals
        spoiler_labels -- new labels to be assigned into spoiled signals. If None, no labels will be changed.

        """
        super().__init__()

        self.spoilers = spoilers
        self.spoilers_weights = spoilers_weights
        self.spoiled_fraction = spoiled_fraction
        self.spoiler_relabelers = spoiler_relabalers

    def fit(self, raw_signals: RawSignals):

        for spoiler in self.spoilers:
            spoiler.fit(raw_signals)

        return self

    def transform(self, raw_signals: RawSignals):

        n_orig_objects = len(raw_signals)

        if self.spoiled_fraction == 1.0:
            sampled_data = deepcopy(raw_signals)
        elif np.isclose(self.spoiled_fraction, 0):

            spoiled_signals = RawSignals(sample_rate=raw_signals.sample_rate)
            return spoiled_signals

        elif self.spoiled_fraction < 1.0:
            indices = np.random.choice(
                [*range(len(raw_signals))],
                size=int(np.ceil(self.spoiled_fraction * n_orig_objects)),
                replace=False,
            )
            sampled_data = raw_signals[indices]
        else:
            indices = np.random.choice(
                [*range(len(raw_signals))],
                size=int(np.ceil(self.spoiled_fraction * n_orig_objects)),
                replace=True,
            )
            sampled_data = raw_signals[indices]

        n_spoilers = len(self.spoilers)
        n_samples = len(sampled_data)
        spoiler_indices = [*range(n_spoilers)]

        sample_spoiler_assign = np.random.choice(
            spoiler_indices, size=n_samples, p=self.spoilers_weights
        )

        spoiled_signals = RawSignals(sample_rate=raw_signals.sample_rate)

        for spoiler_index in spoiler_indices:
            samples_idxs = np.argwhere(sample_spoiler_assign == spoiler_index).flatten()

            spoiled = self.spoilers[spoiler_index].transform(sampled_data[samples_idxs])

            if self.spoiler_relabelers is not None:
                new_labels = self.spoiler_relabelers[spoiler_index].fit_transform(
                    spoiled.get_labels()
                )
                spoiled.set_labels(new_labels)

            spoiled_signals += spoiled

        return spoiled_signals
