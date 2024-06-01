from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

MEANS_DICT = dict()


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a
    single DataFrame or a Tuple[DataFrame, Series]
    """
    global MEANS_DICT
    is_test = y is None

    # combine in order to remove irrelevant data also from (price)
    df = X if is_test else pd.concat([X, y], axis=1)

    # remove irrelevant features - also remove from test since the model
    # is not based on this features
    irrelevant_features = ["id", "lat", "long", "date"]
    df = df.drop(columns=irrelevant_features)

    if is_test:
        for feature in df:
            df[feature] = df[feature].fillna(MEANS_DICT[feature])

    # filter invalid data - only if it's training set
    if not is_test:
        df = df.dropna()
        df = df.drop_duplicates()

        for feature in ["floors", "sqft_living", "sqft_lot"]:
            df = df[df[feature] > 0]

        for feature in ["bedrooms", "bathrooms", "sqft_basement"]:
            df = df[df[feature] >= 0]

        # remove data that isn't based on the values that we expect model
        # to predict
        df = df[(df["floors"].between(0, 4) &
                 df["bathrooms"].between(0, 4) &
                 df["view"].isin(np.arange(5)) &
                 df["waterfront"].isin(np.arange(2)) &
                 df["condition"].isin(np.arange(1, 6)) &
                 df["grade"].isin(np.arange(1, 14)) &
                 ((df["yr_renovated"] == 0) |
                  (df["yr_renovated"] >= df["yr_built"])))]

        MEANS_DICT = {feature: df[feature].mean() for feature in df}

    # create dummy variables to zipcode to both train and test
    zip_dummies = pd.get_dummies(df['zipcode'],
                                 columns=({"zipcode": df["zipcode"].unique()}),
                                 prefix="zipcode: ", dtype=int)
    df = pd.concat([df, zip_dummies], axis=1)
    df = df.drop(columns=['zipcode'])

    if is_test:
        return df

    return df.drop(columns=[y.name]), df[y.name]


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        if "zip" not in feature:
            x = X[feature]
            cov_x_y = np.cov(x, y)[1, 0]  # cov(x,y) in matrix
            std_x = np.std(x)
            std_y = np.std(y)
            pearson_corr = np.round(cov_x_y / (std_x * std_y), decimals=3)
            dataframe = pd.DataFrame({feature: X[feature], 'y': y})
            pc_graph = px.scatter(dataframe, x=feature, y='y')
            pc_graph.update_layout(xaxis_title=f"{feature}",
                                   yaxis_title="Price Value",
                                   title=f"The Correlation between {feature} "
                                         f"and Price. Pearson Correlation: "
                                         f"{pearson_corr}")
            pc_graph.write_image(output_path +
                                 f"/pearson_corr_{feature}_price.png",
                                 width=1200, height=800)


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    df = df[df['price'] > 0]

    # Question 1 - split data into train and test sets
    house_prices = df['price']
    df = df.drop('price', axis=1)
    train_X, train_y, test_X, test_y = split_train_test(df, house_prices)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data. For every percentage p in 10%, 11%, ..., 100%,
    # repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon
    # of size (mean-2*std, mean+2*std)

    test_X = preprocess_data(test_X)

    # reindex columns to synchronize between train and test
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

    # fit test loss over model
    samples_matrix = np.zeros((91, 10))
    percentage = [i for i in range(10, 101)]
    lr = LinearRegression(include_intercept=True)
    for i in range(len(percentage)):
        for j in range(0, 10):
            samples_x = train_X.sample(frac=percentage[i] / 100)
            samples_y = train_y.loc[samples_x.index]
            lr.fit(samples_x, samples_y)
            samples_matrix[i, j] = lr.loss(test_X, test_y)

    avg = samples_matrix.mean(axis=1)
    std = samples_matrix.std(axis=1)

    lower_bound = avg - 2 * std
    upper_bound = avg + 2 * std

    upper_scatter = go.Scatter(x=percentage, y=upper_bound, mode='lines',
                               name='Upper Bound',
                               line=dict(color='grey'))

    lower_scatter = go.Scatter(x=percentage, y=lower_bound, mode='lines',
                               name='Lower Bound',
                               line=dict(color='grey'))

    avg_scatter = go.Scatter(x=percentage, y=avg, mode='lines+markers',
                             name='Average',
                             line=dict(color='deepskyblue'),
                             marker=dict(color='steelblue'))

    upper_lower_between_area = go.Scatter(
        x=np.concatenate([percentage, percentage[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself', fillcolor='rgba(211,211,211,0.5)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False)

    graph = go.Figure(data=[upper_scatter, lower_scatter, avg_scatter,
                            upper_lower_between_area])

    graph.update_layout(xaxis_title="Percentage of Training Samples",
                        yaxis_title="Loss Function Value",
                        title="MSE as Function a of the Training "
                              "Set Percentage Size")

    graph.write_image("MSE_of_percentages_of_samples_size.png",
                      width=1200, height=800)
