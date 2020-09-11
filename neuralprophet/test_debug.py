#!/usr/bin/env python

import unittest
import os
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
import pdb

# for running unit tests with symlinks from .git/hooks
if os.path.islink(__file__):
    old_cwd = os.getcwd()
    cwd = os.path.join(os.path.sep, *os.path.realpath(__file__).split(os.path.sep)[:-1])
    os.chdir(cwd)

from neuralprophet.neural_prophet import NeuralProphet

class UnitTests(unittest.TestCase):
    verbose = True
    plot_verbose = True

    def test_names(self):
        m = NeuralProphet(verbose=verbose)
        m._validate_column_name("hello_friend")


    def test_train_eval_test(self):
        m = NeuralProphet(
            n_lags=14,
            n_forecasts=7,
            verbose=verbose,
            ar_sparsity=0.1,
        )
        df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
        df_train, df_test = m.split_df(df, valid_p=0.1, inputs_overbleed=True)

        metrics = m.fit(df_train, validate_each_epoch=True, valid_p=0.1)
        val_metrics = m.test(df_test)
        if self.verbose:
            print("Metrics: train/eval")
            print(metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))
            print("Metrics: test")
            print(val_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))


    def test_trend(self):
        df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
        m = NeuralProphet(
            n_changepoints=100,
            trend_smoothness=2,
            # trend_threshold=False,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            verbose=verbose,
        )
        m.fit(df)
        future = m.compose_prediction_df(df, future_periods=60, n_history=len(df))
        forecast = m.predict(df=future)
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        if plot:
            plt.show()


    def test_ar_net(self):
        df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
        m = NeuralProphet(
            verbose=verbose,
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
        m.set_forecast_in_focus(m.n_forecasts)
        m.fit(df, validate_each_epoch=True)
        future = m.compose_prediction_df(df, n_history=len(df))
        forecast = m.predict(df=future)
        m.plot_last_forecast(forecast, include_previous_n=3)
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        if self.plot_verbose:
            plt.show()


    def test_seasons(self):
        df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
        # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
        m = NeuralProphet(
            verbose=verbose,
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
        future = m.compose_prediction_df(df, n_history=len(df), future_periods=365)
        forecast = m.predict(df=future)

        if verbose:
            print(sum(abs(m.model.season_params["yearly"].data.numpy())))
            print(sum(abs(m.model.season_params["weekly"].data.numpy())))
            print(m.model.season_params.items())
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        if plot_verbose:
            plt.show()


    def test_lag_reg(self):
        df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
        m = NeuralProphet(
            verbose=verbose,
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
            df['C'] = df['y'].rolling(30, min_periods=1).mean()
            m = m.add_covariate(name='A')
            m = m.add_regressor(name='B')
            m = m.add_regressor(name='C')
            # m.set_forecast_in_focus(m.n_forecasts)
        m.fit(df, validate_each_epoch=True)
        future = m.compose_prediction_df(df, n_history=365)
        forecast = m.predict(future)

        if verbose:
            print(forecast.to_string())
        m.plot_last_forecast(forecast, include_previous_n=10)
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        if plot_verbose:
            plt.show()


    def test_holidays(self):
        df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
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
            verbose=verbose,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        # set event windows
        m = m.add_events(["superbowl", "playoff"], lower_window=-1, upper_window=1, mode="additive")

        # add the country specific holidays
        m = m.add_country_holidays("US", mode="multiplicative")

        history_df = m.create_df_with_events(df, events_df)
        m.fit(history_df)

        # create the test data
        history_df = m.create_df_with_events(df.iloc[100: 500, :].reset_index(drop=True), events_df)
        future = m.compose_prediction_df(df=history_df, events_df=events_df, future_periods=20, n_history=3)
        forecast = m.predict(df=future)
        if verbose:
            print(m.model.event_params)
        m.plot_components(forecast)
        m.plot(forecast)
        m.plot_parameters()
        if plot_verbose:
            plt.show()


    def test_predict(self):
        df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
        m = NeuralProphet(
            verbose=self.verbose,
            n_forecasts=3,
            n_lags=5,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        m.fit(df)
        future = m.compose_prediction_df(df, future_periods=None, n_history=10)
        forecast = m.predict(future)
        # assert False
        m.plot_last_forecast(forecast, include_previous_n=10)
        m.plot(forecast)
        m.plot_components(forecast, crop_last_n=365)
        m.plot_parameters()
        if plot_verbose:
            plt.show()

    def test_plot():
        df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
        m = NeuralProphet(
            verbose=verbose,
            n_forecasts=7,
            n_lags=5,
            yearly_seasonality=8,
            weekly_seasonality=4,
            daily_seasonality=False,
        )
        m.fit(df)
        # m.set_forecast_in_focus(1)
        future = m.compose_prediction_df(df, n_history=1)
        forecast = m.predict(future)
        # m.plot_last_forecast(forecast)
        m.plot(forecast)
        m.plot_components(forecast)
        # m.plot_parameters()
        if plot_verbose:
            plt.show()


if __name__ == '__main__':
    # TODO
    # NOT WORKING test merge hook
    # add argparse to allow for plotting with tests using command line
    # ask Oskar about adding hard performance criteria to training tests, setting seeds
    # test setup.py from scratch with new virtual env

    # for running unit tests from .git/hooks
    if os.path.join('.git', 'hooks') in __file__ or 'pre-commit' in __file__:
        UnitTests.verbose = False
        UnitTests.plot_verbose = False
        tests = UnitTests()
        # run all tests
        results = unittest.main(exit=False)

        if results.result.failures:
            print('Unit tests not passed.')
            print('Exiting... use --no-verify option with git commit to override unit test hook.')
            import sys
            sys.exit(1)

        os.chdir(old_cwd)

    else:
        # to run tests without print output and plotting respectively, default to True
        # UnitTests.verbose = False
        # UnitTests.plot_verbose = False
        tests = UnitTests()

        # to run all tests
        unittest.main(exit=False)

        # to run individual tests
        # test cases: predict (on fitting data, on future data, on completely new data), train_eval, test function, get_last_forecasts, plotting
        # tests.test_names()
        # tests.test_train_eval_test()
        # tests.test_trend()
        # tests.test_seasons()
        # tests.test_ar_net()
        # tests.test_lag_reg()
        # tests.test_holidays()
        # tests.test_predict()
