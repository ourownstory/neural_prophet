#!/usr/bin/env python3

import unittest
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import logging
from neuralprophet import NeuralProphet
import torch
from torch import nn
import numpy as np

log = logging.getLogger("nprophet.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "example_data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
EPOCHS = 5


class IntegrationTests(unittest.TestCase):
    plot = False

    def test_names(self):
        log.info("testing: names")
        m = NeuralProphet()
        m._validate_column_name("hello_friend")

    def test_train_eval_test(self):
        log.info("testing: Train Eval Test")
        m = NeuralProphet(
            n_lags=14,
            n_forecasts=7,
            ar_sparsity=0.1,
            epochs=EPOCHS,
        )
        df = pd.read_csv(PEYTON_FILE)
        df_train, df_test = m.split_df(df, valid_p=0.1, inputs_overbleed=True)

        metrics = m.fit(df_train, freq="D", validate_each_epoch=True, valid_p=0.1)
        val_metrics = m.test(df_test)
        log.debug("Metrics: train/eval: \n {}".format(metrics.to_string(float_format=lambda x: "{:6.3f}".format(x))))
        log.debug("Metrics: test: \n {}".format(val_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x))))

    def test_trend(self):
        log.info("testing: Trend")
        df = pd.read_csv(PEYTON_FILE)
        m = NeuralProphet(
            growth="linear",
            n_changepoints=100,
            changepoints_range=0.9,
            trend_reg=1,
            trend_reg_threshold=False,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=EPOCHS,
        )
        metrics_df = m.fit(df, freq="D")
        future = m.make_future_dataframe(df, future_periods=60, n_historic_predictions=len(df))
        forecast = m.predict(df=future)
        if self.plot:
            m.plot(forecast)
            # m.plot_components(forecast)
            m.plot_parameters()
            plt.show()

    def test_no_trend(self):
        log.info("testing: No-Trend")
        df = pd.read_csv(PEYTON_FILE)
        m = NeuralProphet(
            growth="off",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=EPOCHS,
        )
        # m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
        metrics_df = m.fit(df, freq="D", validate_each_epoch=True)
        future = m.make_future_dataframe(df, future_periods=60, n_historic_predictions=60)

        forecast = m.predict(df=future)
        if self.plot:
            m.plot(forecast)
            m.plot_components(forecast)
            m.plot_parameters()
            plt.show()

    def test_seasons(self):
        log.info("testing: Seasonality")
        df = pd.read_csv(PEYTON_FILE)
        # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
        m = NeuralProphet(
            yearly_seasonality=8,
            weekly_seasonality=4,
            # daily_seasonality=False,
            seasonality_mode="additive",
            # seasonality_mode="multiplicative",
            seasonality_reg=1,
            epochs=EPOCHS,
        )
        metrics_df = m.fit(df, freq="D", validate_each_epoch=True)
        future = m.make_future_dataframe(df, n_historic_predictions=len(df), future_periods=365)
        forecast = m.predict(df=future)
        log.debug("SUM of yearly season params: {}".format(sum(abs(m.model.season_params["yearly"].data.numpy()))))
        log.debug("SUM of weekly season params: {}".format(sum(abs(m.model.season_params["weekly"].data.numpy()))))
        log.debug("season params: {}".format(m.model.season_params.items()))

        if self.plot:
            m.plot(forecast)
            # m.plot_components(forecast)
            m.plot_parameters()
            plt.show()

    def test_custom_seasons(self):
        log.info("testing: Custom Seasonality")
        df = pd.read_csv(PEYTON_FILE)
        # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
        other_seasons = False
        m = NeuralProphet(
            yearly_seasonality=other_seasons,
            weekly_seasonality=other_seasons,
            daily_seasonality=other_seasons,
            seasonality_mode="additive",
            # seasonality_mode="multiplicative",
            seasonality_reg=1,
            epochs=EPOCHS,
        )
        m = m.add_seasonality(name="biannual", period=730, fourier_order=5)
        log.debug("seasonalities: {}".format(m.season_config.periods))
        metrics_df = m.fit(df, freq="D", validate_each_epoch=True)
        future = m.make_future_dataframe(df, n_historic_predictions=len(df), future_periods=30)
        forecast = m.predict(df=future)
        log.debug("season params: {}".format(m.model.season_params.items()))

        if self.plot:
            m.plot(forecast)
            # m.plot_components(forecast)
            m.plot_parameters()
            plt.show()

    def test_ar_net(self):
        log.info("testing: AR-Net")
        df = pd.read_csv(PEYTON_FILE)
        m = NeuralProphet(
            n_forecasts=7,
            n_lags=14,
            # ar_sparsity=0.01,
            # num_hidden_layers=0,
            num_hidden_layers=2,
            d_hidden=64,
            # yearly_seasonality=False,
            # weekly_seasonality=False,
            # daily_seasonality=False,
            epochs=EPOCHS,
        )
        m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
        metrics_df = m.fit(df, freq="D", validate_each_epoch=True)
        future = m.make_future_dataframe(df, n_historic_predictions=len(df) - m.n_lags)
        forecast = m.predict(df=future)
        if self.plot:
            m.plot_last_forecast(forecast, include_previous_forecasts=3)
            m.plot(forecast)
            m.plot_components(forecast)
            m.plot_parameters()
            plt.show()

    def test_lag_reg(self):
        log.info("testing: Lagged Regressors")
        df = pd.read_csv(PEYTON_FILE)
        m = NeuralProphet(
            n_forecasts=3,
            n_lags=7,
            ar_sparsity=0.1,
            # num_hidden_layers=2,
            # d_hidden=64,
            # yearly_seasonality=False,
            # weekly_seasonality=False,
            # daily_seasonality=False,
            epochs=EPOCHS,
        )
        if m.n_lags > 0:
            df["A"] = df["y"].rolling(7, min_periods=1).mean()
            df["B"] = df["y"].rolling(30, min_periods=1).mean()
            m = m.add_lagged_regressor(name="A")
            m = m.add_lagged_regressor(name="B", only_last_value=True)

            # m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
        metrics_df = m.fit(df, freq="D", validate_each_epoch=True)
        future = m.make_future_dataframe(df, n_historic_predictions=365)
        forecast = m.predict(future)

        if self.plot:
            # print(forecast.to_string())
            m.plot_last_forecast(forecast, include_previous_forecasts=10)
            m.plot(forecast)
            m.plot_components(forecast)
            m.plot_parameters()
            plt.show()

    def test_events(self):
        log.info("testing: Events")
        df = pd.read_csv(PEYTON_FILE)
        playoffs = pd.DataFrame(
            {
                "event": "playoff",
                "ds": pd.to_datetime(
                    [
                        "2008-01-13",
                        "2009-01-03",
                        "2010-01-16",
                        "2010-01-24",
                        "2010-02-07",
                        "2011-01-08",
                        "2013-01-12",
                        "2014-01-12",
                        "2014-01-19",
                        "2014-02-02",
                        "2015-01-11",
                        "2016-01-17",
                        "2016-01-24",
                        "2016-02-07",
                    ]
                ),
            }
        )
        superbowls = pd.DataFrame(
            {
                "event": "superbowl",
                "ds": pd.to_datetime(["2010-02-07", "2014-02-02", "2016-02-07"]),
            }
        )
        events_df = pd.concat((playoffs, superbowls))

        m = NeuralProphet(
            n_lags=5,
            n_forecasts=30,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=EPOCHS,
        )
        # set event windows
        m = m.add_events(
            ["superbowl", "playoff"], lower_window=-1, upper_window=1, mode="multiplicative", regularization=0.5
        )

        # add the country specific holidays
        m = m.add_country_holidays("US", mode="additive", regularization=0.5)

        history_df = m.create_df_with_events(df, events_df)
        metrics_df = m.fit(history_df, freq="D")

        # create the test data
        history_df = m.create_df_with_events(df.iloc[100:500, :].reset_index(drop=True), events_df)
        future = m.make_future_dataframe(
            df=history_df, events_df=events_df, future_periods=30, n_historic_predictions=3
        )
        forecast = m.predict(df=future)
        log.debug("Event Parameters:: {}".format(m.model.event_params))
        if self.plot:
            m.plot_components(forecast)
            m.plot(forecast)
            m.plot_parameters()
            plt.show()

    def test_future_reg(self):
        log.info("testing: Future Regressors")
        df = pd.read_csv(PEYTON_FILE)
        m = NeuralProphet(
            n_forecasts=1,
            n_lags=0,
            epochs=EPOCHS,
        )

        df["A"] = df["y"].rolling(7, min_periods=1).mean()
        df["B"] = df["y"].rolling(30, min_periods=1).mean()

        m = m.add_future_regressor(name="A", regularization=0.5)
        m = m.add_future_regressor(name="B", mode="multiplicative", regularization=0.3)

        metrics_df = m.fit(df, freq="D")
        regressors_df = pd.DataFrame(data={"A": df["A"][:50], "B": df["B"][:50]})
        future = m.make_future_dataframe(
            df=df, regressors_df=regressors_df, n_historic_predictions=10, future_periods=50
        )
        forecast = m.predict(df=future)

        if self.plot:
            # print(forecast.to_string())
            # m.plot_last_forecast(forecast, include_previous_forecasts=3)
            m.plot(forecast)
            m.plot_components(forecast)
            m.plot_parameters()
            plt.show()

    def test_predict(self):
        log.info("testing: Predict")
        df = pd.read_csv(PEYTON_FILE)
        m = NeuralProphet(
            n_forecasts=3,
            n_lags=5,
            epochs=EPOCHS,
        )
        metrics_df = m.fit(df, freq="D")
        future = m.make_future_dataframe(df, future_periods=None, n_historic_predictions=len(df) - m.n_lags)
        forecast = m.predict(future)
        if self.plot:
            m.plot_last_forecast(forecast, include_previous_forecasts=10)
            m.plot(forecast)
            m.plot_components(forecast)
            m.plot_parameters()
            plt.show()

    def test_plot(self):
        log.info("testing: Plotting")
        df = pd.read_csv(PEYTON_FILE)
        m = NeuralProphet(
            n_forecasts=7,
            n_lags=14,
            # yearly_seasonality=8,
            # weekly_seasonality=4,
            epochs=EPOCHS,
        )
        metrics_df = m.fit(df, freq="D")

        m.highlight_nth_step_ahead_of_each_forecast(7)
        future = m.make_future_dataframe(df, n_historic_predictions=10)
        forecast = m.predict(future)
        m.plot(forecast)
        m.plot_last_forecast(forecast, include_previous_forecasts=10)
        m.plot_components(forecast)
        m.plot_parameters()

        m.highlight_nth_step_ahead_of_each_forecast(None)
        future = m.make_future_dataframe(df, n_historic_predictions=10)
        forecast = m.predict(future)
        m.plot(forecast)
        m.plot_last_forecast(forecast, include_previous_forecasts=10)
        m.plot_components(forecast)
        m.plot_parameters()
        if self.plot:
            plt.show()

    def test_air_data(self):
        log.info("TEST air_passengers.csv")
        df = pd.read_csv(AIR_FILE)
        m = NeuralProphet(
            n_changepoints=0,
            # trend_reg=1,
            yearly_seasonality=2,
            # seasonality_reg=1,
            # seasonality_mode="additive",
            seasonality_mode="multiplicative",
        )
        metrics = m.fit(df, freq="MS")
        future = m.make_future_dataframe(df, future_periods=48, n_historic_predictions=len(df) - m.n_lags)
        forecast = m.predict(future)
        m.plot(forecast)
        # m.plot_components(forecast)
        m.plot_parameters()
        if self.plot:
            plt.show()

    def test_logistic_trend(self):
        log.info("testing: Logistic growth trend")

        t_min = 0
        t_max = 1
        samples = 3200
        n_changepoints = 5

        ds_freq = "H"

        idx = pd.date_range('2018-01-01', periods=samples, freq=ds_freq)

        t_datetime = pd.Series(idx)
        t = torch.linspace(0, 1, samples)

        changepoints_ds = np.linspace(idx[0].value, idx[-1].value, n_changepoints + 2)[1:-1]
        changepoints_ds = pd.to_datetime(changepoints_ds)

        snr = 5
        torch.manual_seed(5)

        def coeff_determination(y, f):
            '''
            Computes the coefficient of determination of f modeling y
            y: 1D array-like of floats giving outcomes to predict
            f: 1D array-like of floats modeling corresponding values of y
            Returns:
            float, coefficient of determination
            '''
            y_bar = np.mean(y)
            sum_sq_tot = np.sum((y - y_bar) ** 2)
            sum_sq_reg = np.sum((y - f) ** 2)
            return 1 - sum_sq_reg/sum_sq_tot
        
        coeffs_determination = []

        series_proportion = 0.6
        # index of time before which model has access to. Times with indices at or after this time are not trained on.
        current_time_idx = int(len(t) * series_proportion)
        train_t = t[:current_time_idx]
        train_out = train_t
        current_time = t_min + (t_max-t_min) * series_proportion

        # target curves for testing:
        # 1. logistic curve up and down, cap/floor of model given (as in Prophet)
        # 2. smooth logistic curve (as in Prophet)
        # 3. same logistic curve as 3. with learned cap and floor (and with small regularization)
        trend_caps = [[50.0], [5.0], [5.0]]
        trend_floors = [[5.0], [-25.0], [-25.0]]
        trend_k0s = [[24.5123], [100.0], [100.0]]
        trend_deltas = [[12.2064, 0.0,  -150.0,  49.1343,  -9.3666],
                        [12.2064, -25.0,  -160.0,  49.1343,  -9.3666],
                        [12.2064, -25.0,  -160.0,  49.1343,  -9.3666],]
        trend_m0s = [[0.2], [0.2], [0.2]]
        # whether to use target as cap/floor for testing user-set cap/floor
        prespecified_trend_cap = [True, True, False]
        prespecified_trend_floor = [True, True, False]
        n_epochs = [40, 40, 40]
        trend_regs = [0, 0, 0.003]

        runs = len(trend_caps)

        for run in range(runs):
            # create simple logistic growth target trends with additive white noise for testing
            target = NeuralProphet(
                growth='logistic',
                n_changepoints=n_changepoints,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            target.model = target._init_model()
            target.model.trend_cap = nn.Parameter(torch.Tensor(trend_caps[run]))
            target.model.trend_floor = nn.Parameter(torch.Tensor(trend_floors[run]))
            target.model.trend_k0 = nn.Parameter(torch.Tensor(trend_k0s[run]))
            target.model.trend_deltas = nn.Parameter(torch.Tensor(trend_deltas[run]))
            target.model.trend_m0 = nn.Parameter(torch.Tensor(trend_m0s[run]))
            while target.model.trend_m0 > current_time:
                target.model.trend_m0 = nn.Parameter(torch.distributions.normal.Normal(t_min + (t_max-t_min)/2, (t_max - t_min)/4).sample([1]))

            # add white noise
            with torch.no_grad():
                target_trend = target.model._logistic_growth_trend(t)
                noise_sigma = target_trend.std() / snr
                torch.manual_seed(run)
                noisy_target_trend = target_trend + torch.distributions.normal.Normal(0, noise_sigma).sample([len(t)])

            df = pd.DataFrame()
            df['ds'] = t_datetime
            df['y'] = noisy_target_trend
            if prespecified_trend_floor[run]:
                df['floor'] = np.ones_like(target_trend) * target.model.trend_floor.detach().numpy()
            if prespecified_trend_cap[run]:
                df['cap'] = np.ones_like(target_trend) * target.model.trend_cap.detach().numpy()

            model = NeuralProphet(
                growth='logistic',
                trend_reg=trend_regs[run],
                loss_func='l2',
                learning_rate=0.01,
                n_changepoints=n_changepoints,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                trend_cap_user=prespecified_trend_cap[run],
                trend_floor_user=prespecified_trend_floor[run],
            )
            model.train_config['lr'] = 7e-01
            model.train_config['lr_decay'] = 0.93
            model.train_config['batch'] = 32
            model.train_config['epochs'] = n_epochs[run]
            model.model = model._init_model()
            model.fit(df, ds_freq)

            future = model.make_future_dataframe(df, future_periods=0, n_historic_predictions=len(df))

            pred = model.predict(future)['trend']
            coeffs_determination.append(coeff_determination(noisy_target_trend.detach().numpy(),
                                                            pred))

            future_periods = 60
            future = model.make_future_dataframe(df, future_periods=future_periods, 
                                                 n_historic_predictions=len(df), 
                                                 cap_df=np.ones(future_periods) if prespecified_trend_cap[run] else None,
                                                 floor_df=np.ones(future_periods) if prespecified_trend_floor[run] else None)
            forecast = model.predict(df=future)

        # test basic performance with ideal target functions
        assert np.min(coeffs_determination) > 0.94, \
                                'Optimization with logistic growth trend achieving poor performance:\n' \
                                'min coefficient of determination {}\n' \
                                'mean coefficient of determination {}'.format(np.min(coeffs_determination), np.mean(coeffs_determination))
