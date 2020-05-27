import pandas as pd
from code.bifrost import Bifrost
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

    m = Bifrost(
        n_lags=seasonality,
        n_forecasts=1,
        verbose=False,
    )

    m = m.fit(df_train)

    m = m.test(df_val)
    for stat, value in m.results.items():
        print(stat, value)


def test_predict():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # m = Bifrost(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = Bifrost(
        n_lags=30,
        n_changepoints=10,
        n_forecasts=30,
        verbose=True,
        trend_smoothness=0,
        ar_sparsity=None,
    )
    m.fit(df)
    forecast = m.predict(future_periods=30, freq='D')
    m.plot(forecast)
    m.plot(forecast, highlight_forecast=30, crop_last_n=100)
    m.plot_components(forecast)
    plt.show()
    single_forecast = m.get_last_forecasts(3)
    m.plot(single_forecast)
    plt.show()

def test_trend():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # m = Bifrost(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = Bifrost(
        n_lags=30,
        n_changepoints=100,
        n_forecasts=30,
        verbose=True,
        trend_smoothness=100,
    )
    m.fit(df)
    forecast = m.predict(future_periods=60, freq='D')
    m.plot(forecast)
    m.plot_components(forecast)
    plt.show()


def test_ar_net():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # m = Bifrost(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = Bifrost(
        n_forecasts=30,
        n_lags=60,
        n_changepoints=10,
        verbose=True,
        trend_smoothness=0,
        ar_sparsity=0.1,
        num_hidden_layers=2,
        d_hidden=64,
    )
    m.fit(df)
    forecast = m.predict(future_periods=30, freq='D')
    m.plot(forecast)
    m.plot(forecast, highlight_forecast=30, crop_last_n=100)
    # m.plot_components(forecast)
    plt.show()


def test_seasons():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # m = Bifrost(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = Bifrost(
        n_forecasts=1,
        n_lags=0,
        n_changepoints=4,
        verbose=True,
        trend_smoothness=0,
        ar_sparsity=1,
        num_hidden_layers=0,
        d_hidden=64,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        # seasonality_mode='additive',
        seasonality_mode='multiplicative',
        learnign_rate=0.1,
    )
    m.fit(df)
    forecast = m.predict(future_periods=365, freq='D')
    m.plot(forecast)
    m.plot(forecast, highlight_forecast=1, crop_last_n=365+365)
    # m.plot_components(forecast)
    plt.show()


if __name__ == '__main__':
    # test_1()
    # test_predict()
    # test_trend()
    # test_ar_net()
    test_seasons()
