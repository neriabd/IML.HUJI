import numpy as np
import pandas as pd
from numpy import ndarray
from plotly.graph_objs import Scatter, Figure

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple, List, Union, Dict, Any
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import plotly.colors as colors


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where
    the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss
    values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses: List[float] = []

        def append_to_loss(fit: Perceptron, X_i: ndarray, y_i: int):
            losses.append(fit.loss(X, y))

        Perceptron(callback=append_to_loss).fit(X, y)
        df = pd.DataFrame(losses, range(len(losses)))

        # Plot figure of loss as function of fitting iteration
        graph: Figure = px.line(df)
        graph.update_layout(xaxis_title="Number Of Iterations",
                            yaxis_title="Training Loss Error",
                            title=f"Training Perceptron Error - {n}",
                            showlegend=False)
        graph.write_image(f"Perceptron_Training Error_{n}.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified
    covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and
    gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        # Fit models and predict over training set
        gnb: GaussianNaiveBayes = GaussianNaiveBayes().fit(X, y)
        gnb_predict: ndarray = gnb.predict(X)
        lda: LDA = LDA().fit(X, y)
        lda_predict: ndarray = lda.predict(X)

        num_of_labels: int = len(np.unique(y))

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        # Accuracies
        accuracy_gnb: float = \
            np.round(accuracy(y, gnb_predict), decimals=4) * 100
        accuracy_lda: float = \
            np.round(accuracy(y, lda_predict), decimals=4) * 100

        # Create Subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f'Gaussian Naive Bayes: Accuracy - {accuracy_gnb}%',
            f'LDA: Accuracy - {accuracy_lda}%'))

        color_scale: List[Union[str, Any]] = \
            (colors.qualitative.Dark24 * int(np.ceil(num_of_labels / 3)))[
            :num_of_labels]

        map_label_ind: Dict[Any, int] = {value: index for index, value in
                                         enumerate(np.unique(y))}

        true_y_indexes: ndarray = np.array([map_label_ind[val] for val in y])

        # incase there is more than 3 classes
        symbols_class = np.array(
            ["circle", "x", "diamond"] * int(np.ceil(num_of_labels / 3)))

        # Trace - GNB predict
        # True Classes: Circle - Class 0, X - Class 1, Diamond - Class 2
        # GNB Classes: Blue - Class 0, Pink - Class 1 ,Green - Class 2
        gnb_predict_indexes: ndarray = \
            np.array([map_label_ind[val] for val in gnb_predict])
        trace_gnb: Scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                        marker=dict(size=6,
                                                    color=gnb_predict_indexes,
                                                    symbol=symbols_class[
                                                        true_y_indexes],
                                                    colorscale=color_scale))

        # Trace - LDA predict
        # Create Indexes for Correspond Labels
        lda_predict_ind: ndarray = \
            np.array([map_label_ind[val] for val in lda_predict])
        # LDA Classes: Blue - Class 0, Pink - Class 1 ,Green - Class 2
        trace_lda: Scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                        marker=dict(size=6,
                                                    color=lda_predict_ind,
                                                    symbol=symbols_class[
                                                        true_y_indexes],
                                                    colorscale=color_scale))

        # Add traces
        fig.add_trace(trace_gnb, row=1, col=1)
        fig.add_trace(trace_lda, row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        # GNB - Mean X
        gnb_mean_x: Scatter = go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1],
                                         mode='markers',
                                         marker=dict(symbol='x', color='black',
                                                     size=20))
        # LDA - Mean X
        lda_mean_x: Scatter = go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                         mode='markers',
                                         marker=dict(symbol='x', color='black',
                                                     size=20))

        # Add traces
        fig.add_trace(gnb_mean_x, row=1, col=1)
        fig.add_trace(lda_mean_x, row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(num_of_labels):
            # GNB - Ellipse & LDA - Ellipse
            gnb_ellipse = get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i]))
            lda_ellipse = get_ellipse(lda.mu_[i], lda.cov_)

            # Add traces
            fig.add_trace(gnb_ellipse, row=1, col=1)
            fig.add_trace(lda_ellipse, row=1, col=2)

        # Main plot edit
        title_name: str = f.split('.')[0]
        fig.update_layout(
            title={'text': f'Probabilistic Classifiers {title_name} '
                           f'dataset', 'font': {'color': 'blue'}},
            showlegend=False)

        fig.update_xaxes(title_text="1'st feature value", row=1, col=1)
        fig.update_yaxes(title_text="2'st feature value", row=1, col=1)

        fig.update_xaxes(title_text="1'st feature value", row=1, col=2)
        fig.update_yaxes(title_text="2'st feature value", row=1, col=2)

        fig.write_image(f"gauss_naive_bayes_compared_lda_{title_name}.png",
                        width=1100, height=500)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
