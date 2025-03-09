
from collections import defaultdict
from contextlib import suppress

from itertools import product
import numbers
import time
import warnings

from joblib import Parallel, logger
from joblib.parallel import delayed
from sklearn import clone
from sklearn.base import BaseEstimator, is_classifier
from sklearn.calibration import check_cv, indexable
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import indexable, check_is_fitted, _check_fit_params
from sklearn.model_selection._validation import _warn_or_raise_about_fit_failures, _insert_error_scores

from sklearn.utils.validation import _num_samples
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer


import numpy as np
from traceback import format_exc


class GridSearchCVOneClass2(GridSearchCV):
    
    def __init__(self, estimator: BaseEstimator, param_grid, *, scoring= None, n_jobs= None, refit= True,
                  cv= None, verbose= 0, pre_dispatch= "2*n_jobs", error_score=np.nan, return_train_score= False,
                  test_outlier_generator=None) -> None:
        
        super().__init__(estimator, param_grid, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv,
                          verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score,
                            return_train_score=return_train_score)
        
        self.test_outlier_generator = test_outlier_generator

    def fit(self, X, y = None, *, groups = None, **fit_params):


        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(self._fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
    
    def _fit_and_score(self,
        estimator,
        X,
        y,
        scorer,
        train,
        test,
        verbose,
        parameters,
        fit_params,
        return_train_score=False,
        return_parameters=False,
        return_n_test_samples=False,
        return_times=False,
        return_estimator=False,
        split_progress=None,
        candidate_progress=None,
        error_score=np.nan,
    ):

        
        if not isinstance(error_score, numbers.Number) and error_score != "raise":
            raise ValueError(
                "error_score must be the string 'raise' or a numeric value. "
                "(Hint: if using 'raise', please make sure that it has been "
                "spelled correctly.)"
            )

        progress_msg = ""
        if verbose > 2:
            if split_progress is not None:
                progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
            if candidate_progress and verbose > 9:
                progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

        if verbose > 1:
            if parameters is None:
                params_msg = ""
            else:
                sorted_keys = sorted(parameters)  # Ensure deterministic o/p
                params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
        if verbose > 9:
            start_msg = f"[CV{progress_msg}] START {params_msg}"
            print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

        # Adjust length of sample weights
        fit_params = fit_params if fit_params is not None else {}
        fit_params = _check_fit_params(X, fit_params, train)

        if parameters is not None:
            # clone after setting parameters in case any parameters
            # are estimators (like pipeline steps)
            # because pipeline doesn't clone steps in fit
            cloned_parameters = {}
            for k, v in parameters.items():
                cloned_parameters[k] = clone(v, safe=False)

            estimator = estimator.set_params(**cloned_parameters)

        start_time = time.time()

        

        X_train, y_train = X[train], y[train]

        #Add spoiled signals to the test set
        X_test_pre, y_test_pre = X[test], y[test]
        X_outlier, y_outlier = self.test_outlier_generator.fit_generate(X_test_pre, y_test_pre)

        X_test = np.concatenate((X_test_pre, X_outlier))
        y_test = np.concatenate((y_test_pre, y_outlier))
        
        result = {}
        try:
            if y_train is None:
                estimator.fit(X_train, **fit_params)
            else:
                estimator.fit(X_train, y_train, **fit_params)

        except Exception as ex:
            # Note fit time as time until error
            fit_time = time.time() - start_time
            score_time = 0.0
            if error_score == "raise":
                raise
            elif isinstance(error_score, numbers.Number):
                if isinstance(scorer, dict):
                    test_scores = {name: error_score for name in scorer}
                    if return_train_score:
                        train_scores = test_scores.copy()
                else:
                    test_scores = error_score
                    if return_train_score:
                        train_scores = error_score
            result["fit_error"] = format_exc()
        else:
            result["fit_error"] = None

            fit_time = time.time() - start_time
            test_scores = _score(estimator, X_test, y_test, scorer, error_score)
            score_time = time.time() - start_time - fit_time
            if return_train_score:
                train_scores = _score(estimator, X_train, y_train, scorer, error_score)

        if verbose > 1:
            total_time = score_time + fit_time
            end_msg = f"[CV{progress_msg}] END "
            result_msg = params_msg + (";" if params_msg else "")
            if verbose > 2:
                if isinstance(test_scores, dict):
                    for scorer_name in sorted(test_scores):
                        result_msg += f" {scorer_name}: ("
                        if return_train_score:
                            scorer_scores = train_scores[scorer_name]
                            result_msg += f"train={scorer_scores:.3f}, "
                        result_msg += f"test={test_scores[scorer_name]:.3f})"
                else:
                    result_msg += ", score="
                    if return_train_score:
                        result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                    else:
                        result_msg += f"{test_scores:.3f}"
            result_msg += f" total time={logger.short_format_time(total_time)}"

            # Right align the result_msg
            end_msg += "." * (80 - len(end_msg) - len(result_msg))
            end_msg += result_msg
            print(end_msg)

        result["test_scores"] = test_scores
        if return_train_score:
            result["train_scores"] = train_scores
        if return_n_test_samples:
            result["n_test_samples"] = _num_samples(X_test)
        if return_times:
            result["fit_time"] = fit_time
            result["score_time"] = score_time
        if return_parameters:
            result["parameters"] = parameters
        if return_estimator:
            result["estimator"] = estimator
        return result
    
def _score(estimator, X_test, y_test, scorer, error_score="raise"):
    """Compute the score(s) of an estimator on a given test set.

    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _MultimetricScorer(scorers=scorer, raise_exc=(error_score == "raise"))

    try:
        if y_test is None:
            scores = scorer(estimator, X_test)
        else:
            scores = scorer(estimator, X_test, y_test)
    except Exception:
        if isinstance(scorer, _MultimetricScorer):
            # If `_MultimetricScorer` raises exception, the `error_score`
            # parameter is equal to "raise".
            raise
        else:
            if error_score == "raise":
                raise
            else:
                scores = error_score
                warnings.warn(
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )

    # Check non-raised error messages in `_MultimetricScorer`
    if isinstance(scorer, _MultimetricScorer):
        exception_messages = [
            (name, str_e) for name, str_e in scores.items() if isinstance(str_e, str)
        ]
        if exception_messages:
            # error_score != "raise"
            for name, str_e in exception_messages:
                scores[name] = error_score
                warnings.warn(
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{str_e}",
                    UserWarning,
                )

    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores
