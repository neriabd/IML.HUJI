from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    shuffle_indexes = np.arange(X.shape[0])
    np.random.shuffle(shuffle_indexes)
    X = X[shuffle_indexes]
    y = y[shuffle_indexes]

    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    errors_train = np.zeros(cv)
    errors_validation = np.zeros(cv)

    for i in range(cv):
        train_X = np.concatenate(X_folds[:i] + X_folds[i + 1:])
        train_y_true = np.concatenate(y_folds[:i] + y_folds[i + 1:])
        validation_X = X_folds[i]
        validation_y_true = y_folds[i]

        # fit model - train data set
        fitted_model = estimator.fit(train_X, train_y_true)

        # predictions
        train_y_pred = fitted_model.predict(train_X)
        validation_y_pred = fitted_model.predict(validation_X)

        # calculate errors
        errors_train[i] = scoring(train_y_true, train_y_pred)
        errors_validation[i] = scoring(validation_y_true, validation_y_pred)

    return np.mean(errors_train), np.mean(errors_validation)

