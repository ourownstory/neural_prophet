import pandas as pd
from neuralprophet.neural_prophet import NeuralProphet
import matplotlib.pyplot as plt

def test_names():
    m = NeuralProphet()
    m._validate_column_name("hello_friend")


def test_eval(verbose=True):
    from neuralprophet.df_utils import split_df
    # df = pd.read_csv('../data/example_air_passengers.csv')
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # df.head()
    # print(df.shape)
    m = NeuralProphet(
        n_lags=14,
        n_forecasts=7,
        verbose=verbose,
        ar_sparsity=0.1,
        loss_func='huber',
        # impute_missing=False
    )

    # df_train, df_val = m.split_df(df, valid_p=0.2)
    # train_metrics = m.fit(df_train)
    # val_metrics = m.test(df_val)
    # if verbose:
    #     print("Train Metrics:")
    #     print(train_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))
    #     print("Val Metrics:")
    #     print(val_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))

    metrics = m.fit(df, test_each_epoch=True, valid_p=0.2)
    if verbose:
        print(metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))


def test_trend():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_changepoints=100,
        trend_smoothness=3,
        trend_threshold=False,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        loss_func='huber',
        verbose=True,
    )
    # m.set_forecast_in_focus(m.n_forecasts)
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
        # ar_sparsity=0.1,
        num_hidden_layers=0,
        # num_hidden_layers=2,
        # d_hidden=64,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.set_forecast_in_focus(m.n_forecasts)
    m.fit(df)
    forecast = m.predict()
    if verbose:
        # m.plot_last_forecasts(3)
        m.plot(forecast)
        # m.plot(forecast, crop_last_n=10+m.n_lags+m.n_forecasts)
        m.plot_components(forecast)
        plt.show()


def test_seasons(verbose=True):
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = NeuralProphet(
        verbose=verbose,
        n_forecasts=1,
        n_lags=1,
        # n_changepoints=5,
        # trend_smoothness=0,
        yearly_seasonality=16,
        # weekly_seasonality=4,
        daily_seasonality=False,
        seasonality_mode='additive',
        # seasonality_mode='multiplicative',
        seasonality_reg=10,
        # learning_rate=1,
        # normalize_y=True,
    )
    # m.set_forecast_in_focus(m.n_forecasts)
    m.fit(df)
    forecast = m.predict(future_periods=365)
    print(sum(abs(m.model.season_params["yearly"].data.numpy())))
    print(sum(abs(m.model.season_params["weekly"].data.numpy())))
    print(m.model.season_params.items())
    if verbose:
        m.plot_components(forecast)
        m.plot(forecast)
        # m.plot(forecast, crop_last_n=365+m.n_forecasts)
        plt.show()


def test_lag_reg(verbose=True):
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        verbose=verbose,
        n_forecasts=3,
        n_lags=5,
        # n_changepoints=0,
        # trend_smoothness=2,
        ar_sparsity=0.1,
        num_hidden_layers=2,
        # d_hidden=64,
        # yearly_seasonality=False,
        # weekly_seasonality=False,
        # daily_seasonality=False,
        # impute_missing=False
    )
    if m.n_lags > 0:
        df['A'] = df['y'].rolling(7, min_periods=1).mean()
        df['B'] = df['y'].rolling(30, min_periods=1).mean()
        df['C'] = df['y'].rolling(30, min_periods=1).mean()
        m = m.add_covariate(name='A')
        m = m.add_regressor(name='B')
        m = m.add_regressor(name='C')
        m.set_forecast_in_focus(m.n_forecasts)
    m.fit(df, test_each_epoch=True)
    forecast = m.predict(n_history=10)
    # print(forecast.to_string())
    if verbose:
        # m.plot_last_forecasts(3)
        m.plot(forecast)
        m.plot_components(forecast, crop_last_n=365)
        m.plot_parameters()
        plt.show()

def test_holidays(verbose=True):
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    playoffs = pd.DataFrame({
        'holiday': 'playoff',
        'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                              '2010-01-24', '2010-02-07', '2011-01-08',
                              '2013-01-12', '2014-01-12', '2014-01-19',
                              '2014-02-02', '2015-01-11', '2016-01-17',
                              '2016-01-24', '2016-02-07']),
        'lower_window': -1,
        'upper_window': 1,
    })
    superbowls = pd.DataFrame({
        'holiday': 'superbowl',
        'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
        'lower_window': 0,
        'upper_window': 1,
    })
    holidays = pd.concat((playoffs, superbowls))

    # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)

    m = NeuralProphet(
        verbose=True,
        # n_forecasts=3,
        # n_changepoints=5,
        # trend_smoothness=0,
        n_lags=0,
        yearly_seasonality=3,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        # seasonality_mode='multiplicative',
        # seasonality_reg=10,
        # learning_rate=1,
        # normalize_y=True,
    )

    # add the holidays
    holidays_config = dict({
        'superbowl': (0, 1),
        'playoff': (-1, 1)
    })

    m.add_holidays(holidays_config=holidays_config, country_name='US')
    holidays_df = m.make_dataframe_with_holidays(df, holidays)
    m.fit(holidays_df)

    # create the test range data
    holidays_df = m.make_dataframe_with_holidays(data=df, holidays_df=holidays, future_periods=365)
    # forecast = m.predict(holidays_df, future_periods=60)
    forecast = m.predict(holidays_df, future_periods=365)
    print(sum(abs(m.model.holiday_params.data.numpy())))
    print(m.model.holiday_params)
    if verbose:
        m.plot_components(forecast)
        m.plot(forecast)
        plt.show()

if __name__ == '__main__':
    """
    just used for debugging purposes. 
    should implement proper tests at some point in the future.
    (some test methods might already be deprecated)
    """
    # test_names()
    # test_eval()
    # test_trend()
    # test_ar_net()
    # test_seasons()
    # test_lag_reg()
    test_holidays()

    # test cases: predict (on fitting data, on future data, on completely new data), train_eval, test function, get_last_forecasts, plotting


