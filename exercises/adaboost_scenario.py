import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape 
    (num_samples). num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    max_learners = list(range(1, n_learners + 1))
    train_error = [adaboost.partial_loss(train_X, train_y, i) for i in
                   max_learners]
    test_error = [adaboost.partial_loss(test_X, test_y, i) for i in
                  max_learners]

    train_scatter = go.Scatter(x=max_learners, y=train_error, mode='lines',
                               name='Train Error')
    test_scatter = go.Scatter(x=max_learners, y=test_error, mode='lines',
                              name='Test Error')

    # ns - no noise
    adaboost_graph = go.Figure(data=[train_scatter, test_scatter])
    adaboost_graph.update_layout(
        xaxis_title="Number Of Learners (Iterations)",
        yaxis_title="Misclasification Error",
        title=f"<b>Misclassification Error as a Function of Number of Learners"
              f" - Noise {noise}</b>")
    adaboost_graph.write_image \
        (f"Misclasification_Error_Adaboost_Noise_{noise}.png",
         width=960, height=540)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]

    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T \
           + np.array([-.1, .1])

    # db - decision boundaries
    db_subplots = make_subplots(rows=2, cols=2,
                                subplot_titles=[
                                    f"<b>Weak Learners: {iteration}, "
                                    f"Noise: {noise}<b>" for iteration in T],
                                horizontal_spacing=0.1, vertical_spacing=0.13)

    symbol_labels_test = ["circle" if y == 1 else "square" for y in test_y]

    for i, learners in enumerate(T):
        db_subplots.add_traces([
            decision_surface(lambda X: adaboost.partial_predict(X, learners),
                             lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       marker=dict(color=test_y, symbol=symbol_labels_test),
                       showlegend=False)],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    db_subplots.update_layout(
        title={'text': f'<b>Decision Boundaries Of Ensemble<b>',
               'font': {'color': 'blue'}})

    for i, j in product([1, 2], [1, 2]):
        db_subplots.update_xaxes(title_text="1'st feature value", row=i, col=j)
        db_subplots.update_yaxes(title_text="2'st feature value", row=i, col=j)

    db_subplots.write_image \
        (f"Decision_Boundaries_Noise_{noise}.png",
         width=800, height=800)

    # Question 3: Decision surface_full_ensemble of best performing ensemble
    lowest_test_error_ind = np.argmin(test_error)
    accuracy = np.round((1 - test_error[lowest_test_error_ind]) * 100,
                        decimals=2)
    ensemble_size = lowest_test_error_ind + 1
    surface_full_ensemble = decision_surface(
        lambda X: adaboost.partial_predict(X, ensemble_size),
        lims[0], lims[1], showscale=False)

    scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                         showlegend=False,
                         marker=dict(color=test_y, symbol=symbol_labels_test))

    best_ensemble_graph = go.Figure(data=[surface_full_ensemble, scatter])

    best_ensemble_graph.update_layout(title={
        'text': f"<b>Lowest Error Ensemble, Ensemble Size: {ensemble_size}, "
                f"Accuracy: {accuracy}%, Noise: {noise}<b>",
        'font': {'color': 'blue'}},
        xaxis_title="1'st feature value",
        yaxis_title="2'st feature value",
    )

    best_ensemble_graph.write_image \
        (f"Best_Ensemble_Size_{ensemble_size}_Noise_{noise}.png",
         width=800, height=800)

    # Question 4: Decision surface_full_ensemble with weighted samples
    symbol_labels_train = ["circle" if y == 1 else "square" for y in train_y]

    proporsional_sizes = adaboost.D_ / adaboost.D_.max() * 30

    scatter = go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                         marker=dict(color=train_y, size=proporsional_sizes,
                                     symbol=symbol_labels_train),
                         showlegend=False)

    train_weights_graph = go.Figure(data=[surface_full_ensemble, scatter])

    train_weights_graph.update_layout(title={
        'text': f"<b>Train Set Adaboost Proporsional To Final Weights<b>",
        'font': {'color': 'blue'}},
        xaxis_title="1'st feature value",
        yaxis_title="2'st feature value",
    )

    train_weights_graph.write_image \
        (f"Train_Set_Proporsional_Weights_Noise_{noise}.png",
         width=800, height=800)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
