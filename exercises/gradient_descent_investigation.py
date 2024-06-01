import numpy as np
import pandas as pd

from typing import Tuple, List, Callable, Type
from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from plotly.subplots import make_subplots
from itertools import product


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def fill_subplots(subplots, title, file_name, x_name, y_name):
    subplots.update_layout(title={'text': title, 'font': {'color': 'blue'}}, showlegend=False)

    for i, j in product([1, 2], [1, 2]):
        subplots.update_xaxes(title_text=x_name, row=i, col=j)
        subplots.update_yaxes(title_text=y_name, row=i, col=j)

    subplots.write_image(file_name, width=1600, height=800)


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_lst = []
    weights_lst = []

    def callback(solver, weights, val, grad, t, eta, delta) -> None:
        values_lst.append(val)
        weights_lst.append(weights)

    return callback, values_lst, weights_lst


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    norms, names = [L1, L2], ["L1 Norm", "L2 Norm"]

    for norm, name in zip(norms, names):
        losses = []
        cr_subplots = make_subplots(rows=2, cols=2,
                                    subplot_titles=[f"<b>η: {eta}<b>" for eta in etas],
                                    horizontal_spacing=0.12, vertical_spacing=0.13)

        for i, eta in enumerate(etas):
            callback, values, weights = get_gd_state_recorder_callback()
            GD_last = GradientDescent(learning_rate=FixedLR(eta), callback=callback).fit(
                norm(init), None, None)
            losses.append(norm(GD_last).compute_output())

            # dp - descent path
            dp_graph = plot_descent_path(norm, np.array(weights),
                                         title=f"{name}: "f"Learning Rate = {eta}<b>")
            dp_graph.update_layout(title={'font': {'color': 'blue'}},
                                   xaxis_title="w1",
                                   yaxis_title="w2")
            dp_graph.write_image(f"{name}_{eta}.png", width=800, height=800)

            # cr - convergence rate
            cr_subplots.add_traces([
                go.Scatter(x=np.arange(len(values)), y=values, name=f'eta: {eta}')],
                rows=(i // 2) + 1, cols=(i % 2) + 1)

        fill_subplots(subplots=cr_subplots,
                      title=f"<b>Convergence Rate: {name} as a Function of GD Iteration With Different "
                            f"Learning Rates<b>", file_name=f"Convergence_Rate_{name}_FixedLR.png",
                      x_name="GD Iterations", y_name=f"{name} Norm Value")

        min_loss = np.min(losses)
        learning_rate = etas[np.argmin(losses)]
        print(f"Lowest Loss {name}: {min_loss}, Learning Rate: {learning_rate}")


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    cr_subplots = make_subplots(rows=2, cols=2,
                                subplot_titles=[
                                    f"<b>ɣ: {gamma}<b>" for gamma in gammas],
                                horizontal_spacing=0.1, vertical_spacing=0.13)

    for i, gamma in enumerate(gammas):
        # Optimize the L1 objective using different decay-rate values of
        # the exponentially decaying learning rate
        L1_norm = L1(weights=init)
        callback, values, weights = get_gd_state_recorder_callback()
        GD_last = GradientDescent(learning_rate=ExponentialLR(eta, gamma),
                                  callback=callback).fit(L1_norm, None, None)

        # Plot algorithm's convergence for the different values of gamma
        cr_subplots.add_traces([
            go.Scatter(x=np.arange(len(values)), y=values, name=f'gamma: {gamma}')],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

        print(f"Exponential L1 Learning Rate - η: {eta}, Gamma: {gamma} - "
              f"Loss: {L1(weights=GD_last).compute_output()}")

        if gamma == 0.95:
            # Plot descent path for gamma=0.95
            fig = plot_descent_path(L1, np.array(weights),
                                    title=f"L1 Norm Exponential Learning Rate, η: {eta}, ɣ: {gamma}")
            fig.update_layout(title={'font': {'color': 'blue'}},
                              xaxis_title="w1",
                              yaxis_title="w2")
            fig.write_image("Descent_Path_95.png", width=800, height=800)

    fill_subplots(subplots=cr_subplots,
                  title=f"<b>GD: Exponential Learning Rate - η: {eta} With different ɣ's Values<b>",
                  file_name=f"Exponential_LR_Convergence_Rate.png",
                  x_name="GD Iterations", y_name=f"L1 Value")


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(
        y_test)

    # Plotting convergence rate of logistic regression over SA heart disease data
    LR = LogisticRegression(penalty="none").fit(X_train, y_train)
    LR_pred = LR.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, LR_pred)

    LR_fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         marker_size=5, marker_color="lightskyblue", name='TPR / FPR',
                         hovertemplate=
                         "<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Logistic Regression Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    LR_fig.write_image(f"Logistic_Regression_FPR_TPR.png",
                       width=1300, height=800)

    alpha_star = np.round(thresholds[np.argmax(tpr - fpr)], decimals=4)
    loss_alpha_star = np.round(LogisticRegression(alpha=alpha_star).fit(X_train, y_train).loss(X_test, y_test)
                               , decimals=4)
    print(f"alpha star - argmax(TPR - FPR): {alpha_star}, Test Loss: {loss_alpha_star}")

    # Fitting l1- and l2-regularized logistic regression models,
    # using cross-validation to specify values
    # of regularization parameter
    norm_types = ["l1", "l2"]
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    LR = lambda lamb, penalty: LogisticRegression(
        solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
        lam=lamb, penalty=penalty, alpha=0.5)

    for norm in norm_types:
        cross_validation = np.zeros(len(lambdas))
        for i, lam in enumerate(lambdas):
            cross_validation[i] = cross_validate(LR(lam, norm), X_train, y_train, misclassification_error)[1]
        lambda_minimizer = lambdas[np.argmin(cross_validation)]
        fitted_LR = LR(lambda_minimizer, norm).fit(X_train, y_train)
        test_error = np.round(fitted_LR.loss(X_test, y_test), decimals=4)
        print(f"Chosen lambda for {norm} Norm: {lambda_minimizer}")
        print(f"Lowest Test Error with {norm} Norm: {test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
