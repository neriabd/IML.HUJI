from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as pe
import pandas as pd

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    random_samples = np.random.normal(mu, sigma, 1000)
    x = UnivariateGaussian().fit(random_samples)
    print('Estimated expectancy and variance: ')
    print(f'({np.round(x.mu_, decimals=3)}, {np.round(x.var_, decimals=3)})')

    # Question 2 - Empirically showing sample mean is consistent
    drawn_samples = [
        np.abs(UnivariateGaussian().fit(random_samples[:i]).mu_ - mu) for i
        in np.arange(10, 1010, 10)]
    dataframe_exp = pd.DataFrame(drawn_samples,
                                 index=np.arange(10, 1010, 10))
    graph_exp = pe.scatter(dataframe_exp)

    graph_exp.update_layout(
        xaxis_title="Number Of Samples",
        yaxis_title="Distance of True and Estimated Expectancy",
        title="Distance Between True and Estimated Expectancy as "
              "a Function of the Sample Size")
    graph_exp.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = x.pdf(random_samples)
    dataframe_pdf = pd.DataFrame(pdfs,
                                 random_samples)
    graph_pdf = pe.scatter(dataframe_pdf,
                           title="Empirical PDF of Fitted Model")
    graph_pdf.update_layout(
        xaxis_title="Sample Value",
        yaxis_title="Estimated PDF")
    graph_pdf.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov_matrix = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    random_samples = np.random.multivariate_normal(mu, cov_matrix, 1000)
    x = MultivariateGaussian().fit(random_samples)
    print(f'Estimated Expectancy: \n{np.round(x.mu_, decimals=3)}')
    print(f'Estimated Cov: \n{np.round(x.cov_, decimals=3)}')

    # Question 5 - Likelihood evaluation
    line_space = np.linspace(-10, 10, 200)
    llh_mat = np.zeros((200, 200))  # log-likelihood matrix
    for i in range(len(llh_mat)):
        for j in range(len(llh_mat[0])):
            ls_mu = np.array([line_space[i], 0, line_space[j], 0]).T
            llh_mat[i][j] = MultivariateGaussian.log_likelihood(ls_mu,
                                                                x.cov_,
                                                                random_samples)

    heat_map = go.Figure(
        data=go.Heatmap(z=llh_mat, x=line_space, y=line_space))
    heat_map.update_layout(title="Log-Likelihood as a function of f1 and f3",
                           yaxis_title="f1 - 1'st coordinate in expectancy",
                           xaxis_title="f3 - 3'rd coordinate in expectancy")
    heat_map.show()

    # Question 6 - Maximum likelihood
    convert_index = np.unravel_index(llh_mat.argmax(), llh_mat.shape)
    f1_index = line_space[convert_index[0]]
    f3_index = line_space[convert_index[1]]
    max_likelihood = np.round((f1_index, f3_index), decimals=3)
    print(f'Maximum Likelihood Indexes f1, f3: ')
    print(max_likelihood)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
