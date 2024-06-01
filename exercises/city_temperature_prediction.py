import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import calendar

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # City Temperature Data Set
    cty_tmp_ds = pd.read_csv(filename,
                             parse_dates=["Date"]).dropna().drop_duplicates()

    # remove values that seem unlikely
    cty_tmp_ds = cty_tmp_ds[cty_tmp_ds["Day"].isin(np.arange(1, 32)) &
                            cty_tmp_ds["Year"].isin(np.arange(1900, 2100)) &
                            (cty_tmp_ds.Temp > -20) & (cty_tmp_ds.Temp < 55)]

    # drop data containing invalid values of country and city
    cty_tmp_ds = cty_tmp_ds.drop(
        cty_tmp_ds[(cty_tmp_ds["Country"].apply(type) != str) |
                   (cty_tmp_ds["City"].apply(type) != str)].index)

    # change type of year so the colors will be discrete
    cty_tmp_ds['Year'] = cty_tmp_ds['Year'].astype(str)

    # Add Column of Day Of Year
    cty_tmp_ds["DayOfYear"] = cty_tmp_ds["Date"].dt.dayofyear

    return cty_tmp_ds


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    d_set_israel = df[df.Country == "Israel"]

    # Create Graph - Correlation Between a  Day of Year To Temperature in IL
    # dof - day of year, temp - temperature
    dof_temp_graph_il = px.scatter(d_set_israel, x="DayOfYear", y="Temp",
                                   color='Year')

    dof_temp_graph_il.update_layout(xaxis_title="Day of Year",
                                    yaxis_title="Temp in Israel",
                                    title="Correlation Between a Day of Year"
                                          " To Temperature in Israel")

    dof_temp_graph_il.write_image("Temp_DayOfYear.png", width=1200,
                                  height=800)

    # Create Graph - std of Temperature as a function of the month

    mth_tmp_graph = px.bar(
        d_set_israel.groupby(["Month"]).Temp.agg(func=np.std, axis=0),
        text_auto='.4',
        color={i: calendar.month_name[i] for i in range(1, 13)})

    mth_tmp_graph.update_layout(xaxis_title="Month",
                                yaxis_title="std of temperature",
                                title="STD of Daily Temperature in Israel In "
                                      "Every Month Over The Years 1995 - 2007",
                                legend_title_text="Month")

    mth_tmp_graph.write_image("STD_Temp_By_Month.png", width=1200, height=800)

    # Question 3 - Mean and Std based on country
    # reset index - make month and country columns in dataset
    grouped_df = df.groupby(["Month", "Country"]).Temp.agg(
        func=[np.mean, np.std]).reset_index()

    mean_std_graph = px.line(grouped_df, x='Month', y='mean', color='Country',
                             error_y='std')

    mean_std_graph.update_layout(xaxis_title="Month",
                                 yaxis_title="Mean of Temperature",
                                 title="Average of Temperatures Over "
                                       "Month per Country")

    mean_std_graph.write_image("Temp_Avg_Countries.png", width=1200,
                               height=800)

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(
        X=d_set_israel["DayOfYear"], y=d_set_israel["Temp"])

    results = np.zeros(10)
    degree = list(range(1, 11))

    for i in range(len(degree)):
        poly_fit = PolynomialFitting(degree[i]).fit(train_X, train_y)
        results[i] = np.round(poly_fit.loss(test_X, test_y), decimals=2)

    print(results)

    deg_df = pd.DataFrame(results, index=degree)
    deg_df["Degree"] = [str(deg) for deg in degree]

    deg_graph = px.bar(deg_df, color="Degree", text_auto='.4')

    deg_graph.update_layout(xaxis_title="Degree of Polynom",
                            yaxis_title="Loss Value",
                            title="Test Error For Polynom Fitting of "
                                  "Degree 1 - 10")

    deg_graph.write_image("Test_Error_Degree.png", width=1200, height=800)

    # Question 5 - Evaluating fitted model on different countries
    poly_fit = PolynomialFitting(k=5).fit(d_set_israel.DayOfYear,
                                          d_set_israel.Temp)

    error_per_country = dict()
    for country in df['Country'].unique():
        if country != 'Israel':
            ds_country = df[df['Country'] == country]  # data set of 1 country
            error = poly_fit.loss(ds_country['DayOfYear'], ds_country['Temp'])
            error_per_country[country] = error

    # index - rows, column name - Error
    # sort by value of error increased order
    # rename axis and move row to different column
    error_dataset = pd.DataFrame.from_dict(error_per_country,
                                           orient='index', columns=['Error']) \
        .rename_axis('Country').sort_values(by='Error',
                                            ascending=True).reset_index()

    error_model_graph = px.bar(error_dataset, color='Country', text_auto='.4')

    error_model_graph.update_layout(xaxis_title="Country",
                                    yaxis_title="Error Value",
                                    title="Error of a Model Derived From "
                                          "Israel Over Other Countries")

    error_model_graph.write_image("Error_Model_Country.png", width=1200,
                                  height=800)
