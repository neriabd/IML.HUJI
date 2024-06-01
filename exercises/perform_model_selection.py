from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso
from itertools import product
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select
    the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) +
    # eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and
    # report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best
    fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the
        algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and
    # testing portions
    X, y = datasets.load_diabetes(return_X_y=True)

    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], \
        X[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization
    # parameter for Ridge and Lasso regressions
    ridges_lambdas = np.linspace(0.0005, 0.7, n_evaluations)
    lasso_lambdas = np.linspace(.001, 1.5, n_evaluations)
    lambdas = [ridges_lambdas, lasso_lambdas]

    model_generators = [lambda x: RidgeRegression(x, include_intercept=True),
                        lambda x: Lasso(alpha=x, max_iter=2500,
                                        fit_intercept=True)]

    ridge_results = np.zeros((n_evaluations, 2))
    lasso_results = np.zeros((n_evaluations, 2))

    results = [ridge_results, lasso_results]

    numbers_of_models = 2

    for i in range(numbers_of_models):
        model = model_generators[i]
        for j, lam in enumerate(lambdas[i]):
            results[i][j] = cross_validate(model(lam), train_X, train_y,
                                           mean_square_error)

    model_names = ['Ridge', 'Lasso']

    ridge_train_scatter = go.Scatter(x=ridges_lambdas, y=ridge_results[:, 0],
                                     name="Train Error: Ridge")
    ridge_validation_scatter = go.Scatter(x=ridges_lambdas,
                                          y=ridge_results[:, 1],
                                          name="Validation Error: Ridge")
    lasso_train_scatter = go.Scatter(x=lasso_lambdas, y=lasso_results[:, 0],
                                     name="Train Error: Lasso")
    lasso_validation_scatter = go.Scatter(x=lasso_lambdas,
                                          y=lasso_results[:, 1],
                                          name="Validation Error: Lasso")

    fold_subplots = make_subplots(rows=1, cols=2,
                                  subplot_titles=[
                                      f"<b>{model} Regression<b>" for model in
                                      model_names])

    fold_subplots.add_traces([ridge_train_scatter, ridge_validation_scatter,
                              lasso_train_scatter, lasso_validation_scatter],
                             rows=[1, 1, 1, 1],
                             cols=[1, 1, 2, 2])

    for i, j in product([1], [1, 2]):
        fold_subplots.update_xaxes(title_text="Lambda", row=i, col=j)
        fold_subplots.update_yaxes(title_text="Mean Square Error", row=i,
                                   col=j)

    fold_subplots.update_layout(
        title={'text': f'<b>Train MSE and Validation MSE Over Cross '
                       f'Validation<b>',
               'font': {'color': 'blue'}})

    fold_subplots.write_image("MSE_Ridge_Lasso.png", width=1500, height=768)

    # Question 3 - Compare best Ridge model, best Lasso model and Least
    # Squares model
    best_lambda_ridge = np.round(
        ridges_lambdas[np.argmin(ridge_results[:, 1])],
        decimals=5)
    print('Ridge - Best Fitting Lambda:', best_lambda_ridge, '\n')

    best_lambda_lasso = np.round(lasso_lambdas[np.argmin(lasso_results[:, 1])],
                                 decimals=5)
    print('Lasso - Best Fitting Lambda:', best_lambda_lasso, '\n')

    ridge = model_generators[0](best_lambda_ridge).fit(train_X, train_y)
    pred_ridge = ridge.predict(test_X)
    error_ridge = np.round(mean_square_error(test_y, pred_ridge), decimals=2)
    print(f'Ridge MSE: {error_ridge}\n')

    lasso = Lasso(best_lambda_lasso).fit(train_X, train_y)
    pred_lasso = lasso.predict(test_X)
    error_lasso = np.round(mean_square_error(test_y, pred_lasso), decimals=2)
    print(f'Lasso MSE: {error_lasso}\n')

    least_squares = LinearRegression().fit(train_X, train_y)
    least_squares_pred = least_squares.predict(test_X)
    error = np.round(mean_square_error(test_y, least_squares_pred), decimals=2)
    print(f"Least Square Regression: Error: {error}\n")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
