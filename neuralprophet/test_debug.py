import pandas as pd
import matplotlib.pyplot as plt
import logging

from neuralprophet.neural_prophet import NeuralProphet
from neuralprophet.utils import set_logger_level

log = logging.getLogger("nprophet.test_debug")
log.setLevel("DEBUG")


def test_names():
    m = NeuralProphet()
    m._validate_column_name("hello_friend")


def test_train_eval_test(plot=False):
    m = NeuralProphet(
        n_lags=14,
        n_forecasts=7,
        ar_sparsity=0.1,
    )
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    df_train, df_test = m.split_df(df, valid_p=0.1, inputs_overbleed=True)

    metrics = m.fit(df_train, validate_each_epoch=True, valid_p=0.1)
    val_metrics = m.test(df_test)
    log.debug("Metrics: train/eval")
    log.debug(metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))
    log.debug("Metrics: test")
    log.debug(val_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))


def test_trend(plot=False):
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_changepoints=100,
        trend_smoothness=2,
        # trend_threshold=False,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.fit(df)
    future = m.make_future_dataframe(df, future_periods=60, n_historic_predictions=len(df))
    forecast = m.predict(df=future)
    if plot:
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_ar_net(plot=False):
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_forecasts=14,
        n_lags=28,
        ar_sparsity=0.01,
        # num_hidden_layers=0,
        num_hidden_layers=2,
        # d_hidden=64,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    m.fit(df, validate_each_epoch=True)
    future = m.make_future_dataframe(df, n_historic_predictions=len(df))
    forecast = m.predict(df=future)
    if plot:
        m.plot_last_forecast(forecast, include_previous_forecasts=3)
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_seasons(plot=False):
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = NeuralProphet(
        # n_forecasts=1,
        # n_lags=1,
        # n_changepoints=5,
        # trend_smoothness=0,
        yearly_seasonality=8,
        weekly_seasonality=4,
        # daily_seasonality=False,
        # seasonality_mode='additive',
        seasonality_mode='multiplicative',
        # seasonality_reg=10,
    )
    m.fit(df, validate_each_epoch=True)
    future = m.make_future_dataframe(df, n_historic_predictions=len(df), future_periods=365)
    forecast = m.predict(df=future)
    log.debug("SUM of yearly season params", sum(abs(m.model.season_params["yearly"].data.numpy())))
    log.debug("SUM of weekly season params", sum(abs(m.model.season_params["weekly"].data.numpy())))
    log.debug("season params", m.model.season_params.items())

    if plot:
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_lag_reg(plot=False):
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_forecasts=3,
        n_lags=5,
        ar_sparsity=0.1,
        # num_hidden_layers=2,
        # d_hidden=64,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    if m.n_lags > 0:
        df['A'] = df['y'].rolling(7, min_periods=1).mean()
        df['B'] = df['y'].rolling(30, min_periods=1).mean()
        m = m.add_lagged_regressor(name='A')
        m = m.add_lagged_regressor(name='B', only_last_value=True)

        # m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    m.fit(df, validate_each_epoch=True)
    future = m.make_future_dataframe(df, n_historic_predictions=365)
    forecast = m.predict(future)

    if plot:
        # print(forecast.to_string())
        m.plot_last_forecast(forecast, include_previous_forecasts=10)
        m.plot(forecast)
        m.plot_components(forecast, figsize=(10, 30))
        m.plot_parameters(figsize=(10,30))
        plt.show()


def test_events(plot=False):
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    playoffs = pd.DataFrame({
        'event': 'playoff',
        'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                              '2010-01-24', '2010-02-07', '2011-01-08',
                              '2013-01-12', '2014-01-12', '2014-01-19',
                              '2014-02-02', '2015-01-11', '2016-01-17',
                              '2016-01-24', '2016-02-07']),
    })
    superbowls = pd.DataFrame({
        'event': 'superbowl',
        'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
    })
    events_df = pd.concat((playoffs, superbowls))

    m = NeuralProphet(
        n_lags=5,
        n_forecasts=3,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    # set event windows
    m = m.add_events(["superbowl", "playoff"], lower_window=-1, upper_window=1, mode="multiplicative", regularization=0.5)

    # add the country specific holidays
    m = m.add_country_holidays("US", mode="additive", regularization=0.5)

    history_df = m.create_df_with_events(df, events_df)
    m.fit(history_df)

    # create the test data
    history_df = m.create_df_with_events(df.iloc[100: 500, :].reset_index(drop=True), events_df)
    future = m.make_future_dataframe(df=history_df, events_df=events_df, future_periods=20, n_historic_predictions=3)
    forecast = m.predict(df=future)
    log.debug("Event Parameters:", m.model.event_params)
    if plot:
        m.plot_components(forecast, figsize=(10, 30))
        m.plot(forecast)
        m.plot_parameters(figsize=(10, 30))
        plt.show()


def test_future_reg(plot=False):
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
    )

    df['A'] = df['y'].rolling(7, min_periods=1).mean()
    df['B'] = df['y'].rolling(30, min_periods=1).mean()

    m = m.add_future_regressor(name='A', regularization=0.5)
    m = m.add_future_regressor(name='B', mode="multiplicative", regularization=0.3)

    m.fit(df)
    regressors_df = pd.DataFrame(data={'A': df['A'][:50], 'B': df['B'][:50]})
    future = m.make_future_dataframe(df=df, regressors_df=regressors_df, n_historic_predictions=10, future_periods=50)
    forecast = m.predict(df=future)

    if plot:
        # print(forecast.to_string())
        # m.plot_last_forecast(forecast, include_previous_forecasts=3)
        m.plot(forecast)
        m.plot_components(forecast, figsize=(10, 30))
        m.plot_parameters(figsize=(10, 30))
        plt.show()


def test_predict(plot=False):
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_forecasts=3,
        n_lags=5,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.fit(df)
    future = m.make_future_dataframe(df, future_periods=None, n_historic_predictions=10)
    forecast = m.predict(future)
    if plot:
        m.plot_last_forecast(forecast, include_previous_forecasts=10)
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_plot(plot=False):
    df = pd.read_csv('../example_data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=14,
        # yearly_seasonality=8,
        # weekly_seasonality=4,
        # daily_seasonality=False,
    )
    m.fit(df)
    m.highlight_nth_step_ahead_of_each_forecast(7)
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    # print(future.to_string())
    # print(forecast.to_string())
    # m.plot_last_forecast(forecast)
    m.plot(forecast)
    m.plot_components(forecast)
    m.plot_parameters()
    if plot:
        plt.show()


def test_logger(plot=False):
    # log.setLevel("ERROR")
    # log.parent.setLevel("WARNING")
    # log.warning("### this WARNING should not show ###")
    # log.parent.warning("this WARNING should show")
    # log.error("this ERROR should show")

    log.setLevel("DEBUG")
    log.parent.setLevel("ERROR")
    log.debug("this DEBUG should show")
    log.parent.warning("### this WARNING not show ###")
    log.error("this ERROR should show")
    log.parent.error("this ERROR should show, too")
    # test existing test cases
    # test_all(log_level="DEBUG")

    # test the set_log_level function
    log.parent.setLevel("INFO")
    m = NeuralProphet(
        n_forecasts=3,
        n_lags=5,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        log_level="DEBUG"
    )
    log.parent.debug("this DEBUG should show")
    m.set_log_level(log_level="WARNING")
    log.parent.debug("### this DEBUG should not show ###")
    log.parent.info("### this INFO should not show ###")


def test_all(log_level=None, global_log_level=None):
    """
    Note: log_level 'NOTSET' shows plots.
    """
    if log_level is not None:
        set_logger_level(log, log_level)
    if global_log_level is not None:
        set_logger_level(log.parent, global_log_level)

    test_names()
    test_train_eval_test()
    test_trend()
    test_ar_net()
    test_seasons()
    test_lag_reg()
    test_events()
    test_predict()
    test_logger()


if __name__ == '__main__':
    """
    just used for debugging purposes. 
    should implement proper tests at some point in the future.
    (some test methods might already be deprecated)
    """
    test_all()
    # test_names(plot=True)
    # test_train_eval_test(plot=True)
    # test_trend(plot=True)
    # test_ar_net(plot=True)
    # test_seasons(plot=True)
    # test_lag_reg(plot=True)
    # test_future_reg(plot=True)
    # test_events(plot=True)
    # test_predict(plot=True)
    # test_plot(plot=True)
    # test_logger(plot=True)
