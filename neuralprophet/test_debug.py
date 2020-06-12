import pandas as pd
from neuralprophet.neural_prophet import NeuralProphet
import matplotlib.pyplot as plt


def test_1():
    df = pd.read_csv('../data/example_air_passengers.csv')

    df.head()
    # print(df.shape)
    seasonality = 12
    train_frac = 0.8
    train_num = int((train_frac * df.shape[0]) // seasonality * seasonality)
    # print(train_num)
    df_train = df.copy(deep=True).iloc[:train_num]
    df_val = df.copy(deep=True).iloc[train_num:]
    m = NeuralProphet(
        n_lags=seasonality,
        n_forecasts=1,
        verbose=False,
    )
    m = m.fit(df_train)
    m = m.test(df_val)
    for stat, value in m.results.items():
        print(stat, value)


def test_trend():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_changepoints=100,
        trend_smoothness=0,
        trend_threshold=True,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        loss_func='huber',
        verbose=True,
    )
    m.fit(df)
    forecast = m.predict(future_periods=60)
    m.plot(forecast)
    m.plot_components(forecast)
    plt.show()


def test_ar_net(verbose=True):
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        verbose=verbose,
        n_forecasts=3,
        n_lags=10,
        n_changepoints=0,
        trend_smoothness=0,
        ar_sparsity=0.1,
        num_hidden_layers=0,
        # num_hidden_layers=2,
        d_hidden=64,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.fit(df)
    forecast = m.predict()
    if verbose:
        # m.plot_last_forecasts(3)
        # m.plot(forecast)
        # m.plot(forecast, highlight_forecast=m.n_forecasts, crop_last_n=10+m.n_lags+m.n_forecasts)
        m.plot_components(forecast, ar_coeff_forecast_n=1)
        plt.show()


def test_seasons(verbose=True):
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = NeuralProphet(
        verbose=verbose,
        n_forecasts=1,
        # n_changepoints=5,
        # trend_smoothness=0,
        yearly_seasonality=1,
        weekly_seasonality=1,
        daily_seasonality=False,
        seasonality_mode='additive',
        # seasonality_mode='multiplicative',
        # learning_rate=1,
        # normalize_y=True,
    )
    m.fit(df)
    forecast = m.predict(future_periods=365)
    if verbose:
        m.plot_components(forecast)
        m.plot(forecast)
        # m.plot(forecast, highlight_forecast=m.n_forecasts, crop_last_n=365+m.n_forecasts)
        plt.show()


if __name__ == '__main__':
    """
    just used for debugging purposes. 
    should implement proper tests at some point in the future.
    (some test methods might already be deprecated)
    """
    # test_1()
    # test_predict()
    test_trend()
    # test_ar_net()
    # test_seasons()
