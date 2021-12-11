#!/usr/bin/env python3

# import pytest
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import torch

from neuralprophet import NeuralProphet, set_random_seed
from neuralprophet import df_utils

log = logging.getLogger("NP.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 512
EPOCHS = 3
BATCH_SIZE = 32

PLOT = False


def test_names():
    log.info("testing: names")
    m = NeuralProphet()
    m._validate_column_name("hello_friend")


def test_train_eval_test():
    log.info("testing: Train Eval Test")
    m = NeuralProphet(
        n_lags=10,
        n_forecasts=3,
        ar_sparsity=0.1,
        epochs=3,
        batch_size=32,
    )
    df = pd.read_csv(PEYTON_FILE, nrows=95)
    df = df_utils.check_dataframe(df, check_y=False)
    df = m.handle_missing_data(df, freq="D", predicting=False)
    df_train, df_test = m.split_df(df, freq="D", valid_p=0.1)
    metrics = m.fit(df_train, freq="D", validation_df=df_test)
    val_metrics = m.test(df_test)
    log.debug("Metrics: train/eval: \n {}".format(metrics.to_string(float_format=lambda x: "{:6.3f}".format(x))))
    log.debug("Metrics: test: \n {}".format(val_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x))))


def test_trend():
    log.info("testing: Trend")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        growth="linear",
        n_changepoints=10,
        changepoints_range=0.9,
        trend_reg=1,
        trend_reg_threshold=False,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    # print(m.config_trend)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=60, n_historic_predictions=60)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot(forecast)
        # m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_custom_changepoints():
    log.info("testing: Custom Changepoints")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    dates = df["ds"][range(1, len(df) - 1, int(len(df) / 5.0))]
    dates_list = [str(d) for d in dates]
    dates_array = pd.to_datetime(dates_list).values
    log.debug("dates: {}".format(dates))
    log.debug("dates_list: {}".format(dates_list))
    log.debug("dates_array: {} {}".format(dates_array.dtype, dates_array))
    for cp in [dates_list, dates_array]:
        m = NeuralProphet(
            changepoints=cp,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
        # print(m.config_trend)
        metrics_df = m.fit(df, freq="D")
        future = m.make_future_dataframe(df, periods=60, n_historic_predictions=60)
        forecast = m.predict(df=future)
        if PLOT:
            # m.plot(forecast)
            # m.plot_components(forecast)
            m.plot_parameters()
            plt.show()


def test_no_trend():
    log.info("testing: No-Trend")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    m = NeuralProphet(
        growth="off",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    # m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=60, n_historic_predictions=60)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_seasons():
    log.info("testing: Seasonality: additive")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = NeuralProphet(
        yearly_seasonality=8,
        weekly_seasonality=4,
        seasonality_mode="additive",
        seasonality_reg=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)
    log.debug("SUM of yearly season params: {}".format(sum(abs(m.model.season_params["yearly"].data.numpy()))))
    log.debug("SUM of weekly season params: {}".format(sum(abs(m.model.season_params["weekly"].data.numpy()))))
    log.debug("season params: {}".format(m.model.season_params.items()))
    if PLOT:
        m.plot(forecast)
        # m.plot_components(forecast)
        m.plot_parameters()
        plt.show()
    log.info("testing: Seasonality: multiplicative")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = NeuralProphet(
        yearly_seasonality=8,
        weekly_seasonality=4,
        seasonality_mode="multiplicative",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)


def test_custom_seasons():
    log.info("testing: Custom Seasonality")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
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
        batch_size=BATCH_SIZE,
    )
    m = m.add_seasonality(name="quarterly", period=90, fourier_order=5)
    log.debug("seasonalities: {}".format(m.season_config.periods))
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)
    log.debug("season params: {}".format(m.model.season_params.items()))
    if PLOT:
        m.PLOT(forecast)
        # m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_ar():
    log.info("testing: AR")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=7,
        yearly_seasonality=False,
        epochs=EPOCHS,
        # batch_size=BATCH_SIZE,
    )
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=90)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot_last_forecast(forecast, include_previous_forecasts=3)
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_ar_sparse():
    log.info("testing: AR (sparse")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=3,
        n_lags=14,
        ar_sparsity=0.5,
        yearly_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=90)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot_last_forecast(forecast, include_previous_forecasts=3)
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_ar_deep():
    log.info("testing: AR-Net (deep)")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=14,
        num_hidden_layers=2,
        d_hidden=32,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=90)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot_last_forecast(forecast, include_previous_forecasts=3)
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_lag_reg():
    log.info("testing: Lagged Regressors")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=2,
        n_lags=3,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    m = m.add_lagged_regressor(names="A")
    m = m.add_lagged_regressor(names="B", only_last_value=True)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    if PLOT:
        print(forecast.to_string())
        m.plot_last_forecast(forecast, include_previous_forecasts=5)
        m.PLOT(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_lag_reg_deep():
    log.info("testing: List of Lagged Regressors (deep)")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=1,
        n_lags=14,
        num_hidden_layers=2,
        d_hidden=32,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(15, min_periods=1).mean()
    df["C"] = df["y"].rolling(30, min_periods=1).mean()
    cols = [col for col in df.columns if col not in ["ds", "y"]]
    m = m.add_lagged_regressor(names=cols)
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    forecast = m.predict(df)
    if PLOT:
        # print(forecast.to_string())
        # m.plot_last_forecast(forecast, include_previous_forecasts=10)
        # m.plot(forecast)
        # m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_events():
    log.info("testing: Events")
    df = pd.read_csv(PEYTON_FILE)[-NROWS:]
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
        n_lags=2,
        n_forecasts=30,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    # set event windows
    m = m.add_events(
        ["superbowl", "playoff"], lower_window=-1, upper_window=1, mode="multiplicative", regularization=0.5
    )
    # add the country specific holidays
    m = m.add_country_holidays("US", mode="additive", regularization=0.5)
    m.add_country_holidays("Indonesia")
    history_df = m.create_df_with_events(df, events_df)
    metrics_df = m.fit(history_df, freq="D")
    future = m.make_future_dataframe(df=history_df, events_df=events_df, periods=30, n_historic_predictions=90)
    forecast = m.predict(df=future)
    log.debug("Event Parameters:: {}".format(m.model.event_params))
    if PLOT:
        m.plot_components(forecast)
        m.PLOT(forecast)
        m.plot_parameters()
        plt.show()


def test_future_reg():
    log.info("testing: Future Regressors")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS + 50)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    regressors_df_future = pd.DataFrame(data={"A": df["A"][-50:], "B": df["B"][-50:]})
    df = df[:-50]
    m = m.add_future_regressor(name="A")
    m = m.add_future_regressor(name="B", mode="multiplicative")
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df=df, regressors_df=regressors_df_future, n_historic_predictions=10, periods=50)
    forecast = m.predict(df=future)
    if PLOT:
        m.PLOT(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_plot():
    log.info("testing: Plotting")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=14,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=m.n_forecasts, n_historic_predictions=10)
    forecast = m.predict(future)
    m.plot(forecast)
    m.plot_last_forecast(forecast, include_previous_forecasts=10)
    m.plot_components(forecast)
    m.plot_parameters()
    m.highlight_nth_step_ahead_of_each_forecast(7)
    forecast = m.predict(df)
    m.plot(forecast)
    m.plot_last_forecast(forecast, include_previous_forecasts=10)
    m.plot_components(forecast)
    m.plot_parameters()
    if PLOT:
        plt.show()


def test_air_data():
    log.info("TEST air_passengers.csv")
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet(
        n_changepoints=0,
        yearly_seasonality=2,
        seasonality_mode="multiplicative",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics = m.fit(df, freq="MS")
    future = m.make_future_dataframe(df, periods=48, n_historic_predictions=len(df) - m.n_lags)
    forecast = m.predict(future)
    if PLOT:
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
        plt.show()


def test_random_seed():
    log.info("TEST random seed")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    set_random_seed(0)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)
    checksum1 = sum(forecast["yhat1"].values)
    set_random_seed(0)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)
    checksum2 = sum(forecast["yhat1"].values)
    set_random_seed(1)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)
    checksum3 = sum(forecast["yhat1"].values)
    log.debug("should be same: {} and {}".format(checksum1, checksum2))
    log.debug("should not be same: {} and {}".format(checksum1, checksum3))
    assert math.isclose(checksum1, checksum2)
    assert not math.isclose(checksum1, checksum3)


def test_yosemite():
    log.info("TEST Yosemite Temps")
    df = pd.read_csv(YOS_FILE, nrows=NROWS)
    m = NeuralProphet(
        changepoints_range=0.95,
        n_changepoints=15,
        weekly_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics = m.fit(df, freq="5min")
    future = m.make_future_dataframe(df, periods=12 * 24, n_historic_predictions=12 * 24)
    forecast = m.predict(future)
    if PLOT:
        m.plot(forecast)
        m.plot_parameters()
        plt.show()


def test_model_cv():
    log.info("CV from model")

    def check_simple(df):
        m = NeuralProphet()
        folds = m.crossvalidation_split_df(df, freq="D", k=5, fold_pct=0.1, fold_overlap_pct=0.5)
        assert all([70 + i * 5 == len(train) for i, (train, val) in enumerate(folds)])
        assert all([10 == len(val) for (train, val) in folds])

    def check_cv(df, freq, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct):
        m = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
        )
        folds = m.crossvalidation_split_df(df, freq=freq, k=k, fold_pct=fold_pct, fold_overlap_pct=fold_overlap_pct)
        total_samples = len(df) - m.n_lags + 2 - (2 * m.n_forecasts)
        per_fold = int(fold_pct * total_samples)
        not_overlap = per_fold - int(fold_overlap_pct * per_fold)
        assert all([per_fold == len(val) - m.n_lags + 1 - m.n_forecasts for (train, val) in folds])
        assert all(
            [
                total_samples - per_fold - (k - i - 1) * not_overlap == len(train) - m.n_lags + 1 - m.n_forecasts
                for i, (train, val) in enumerate(folds)
            ]
        )

    check_simple(pd.DataFrame({"ds": pd.date_range(start="2017-01-01", periods=100), "y": np.arange(100)}))
    check_cv(
        df=pd.DataFrame({"ds": pd.date_range(start="2017-01-01", periods=100), "y": np.arange(100)}),
        n_lags=10,
        n_forecasts=5,
        freq="D",
        k=5,
        fold_pct=0.1,
        fold_overlap_pct=0,
    )
    check_cv(
        df=pd.DataFrame({"ds": pd.date_range(start="2017-01-01", periods=100), "y": np.arange(100)}),
        n_lags=10,
        n_forecasts=15,
        freq="D",
        k=5,
        fold_pct=0.1,
        fold_overlap_pct=0.5,
    )


def test_loss_func():
    log.info("TEST setting torch.nn loss func")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    loss_fn = torch.nn.MSELoss()
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        loss_func=loss_fn,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)


def test_callable_loss():
    log.info("TEST Callable Loss")

    def loss(output, target):
        assym_penalty = 1.25
        beta = 1
        e = target - output
        me = torch.abs(e)
        z = torch.where(me < beta, 0.5 * (me ** 2) / beta, me - 0.5 * beta)
        z = torch.where(e < 0, z, assym_penalty * z)
        return z.mean()

    df = pd.read_csv(YOS_FILE, nrows=NROWS)
    m = NeuralProphet(
        seasonality_mode="multiplicative",
        loss_func=loss,
        changepoints_range=0.95,
        n_changepoints=15,
        weekly_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics = m.fit(df, freq="5min")
    future = m.make_future_dataframe(df, periods=12 * 24, n_historic_predictions=12 * 24)
    forecast = m.predict(future)


def test_custom_torch_loss():
    log.info("TEST PyTorch Custom Loss")

    class Loss(torch.nn.modules.loss._Loss):
        def __init__(self, size_average=None, reduce=None, reduction="mean"):
            super(Loss, self).__init__(size_average, reduce, reduction)

        def forward(self, input, target):
            alpha = 0.9
            y_diff = target - input
            yhat_diff = input - target
            loss = (
                (
                    alpha * torch.max(y_diff, torch.zeros_like(y_diff))
                    + (1 - alpha) * torch.max(yhat_diff, torch.zeros_like(yhat_diff))
                )
                .sum()
                .mean()
            )
            return loss

    loss = Loss()
    df = pd.read_csv(YOS_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics = m.fit(df, freq="5min")
    future = m.make_future_dataframe(df, periods=12, n_historic_predictions=12)
    forecast = m.predict(future)


def test_global_modeling():
    log.debug("Testing: Global Modeling")
    df = pd.read_csv(PEYTON_FILE)
    df1 = df.iloc[:600, :].copy(deep=True)
    df2 = df.iloc[600:1200, :].copy(deep=True)
    df3 = df.iloc[1200:1800, :].copy(deep=True)
    df4 = df.iloc[1800:2400, :].copy(deep=True)
    df1_0 = df1.copy(deep=True)
    df2_0 = df2.copy(deep=True)
    df3_0 = df3.copy(deep=True)
    df4_0 = df4.copy(deep=True)
    df1["A"] = df1["y"].rolling(30, min_periods=1).mean()
    df2["A"] = df2["y"].rolling(10, min_periods=1).mean()
    df3["A"] = df3["y"].rolling(40, min_periods=1).mean()
    df4["A"] = df4["y"].rolling(20, min_periods=1).mean()
    # future_regressors_df1 = pd.DataFrame(data={'A': df1['A'][:25]})
    # future_regressors_df2 = pd.DataFrame(data={'A': df2['A'][:30]})
    future_regressors_df3 = pd.DataFrame(data={"A": df3["A"][:30]})
    future_regressors_df4 = pd.DataFrame(data={"A": df4["A"][:40]})
    playoffs_history = pd.DataFrame(
        {
            "event": "playoff",
            "ds": pd.to_datetime(
                [
                    "2007-12-13",
                    "2009-08-23",
                    "2009-08-25",
                    "2009-09-02",
                    "2011-05-09",
                    "2011-05-11",
                    "2011-05-16",
                    "2013-01-01",
                    "2013-01-05",
                    "2013-01-10",
                    "2014-08-27",
                    "2014-08-31",
                ]
            ),
        }
    )
    history_events_df1 = playoffs_history.iloc[:3, :].copy(deep=True)
    history_events_df2 = playoffs_history.iloc[3:6, :].copy(deep=True)
    history_events_df3 = playoffs_history.iloc[6:9, :].copy(deep=True)
    history_events_df4 = playoffs_history.iloc[9:, :].copy(deep=True)
    playoffs_future = pd.DataFrame(
        {
            "event": "playoff",
            "ds": pd.to_datetime(
                [
                    "2009-08-27",
                    "2009-08-29",
                    "2011-05-14",
                    "2011-05-17",
                    "2013-01-10",
                    "2013-01-15",
                    "2014-09-05",
                    "2014-09-08",
                ]
            ),
        }
    )
    # future_events_df1 = playoffs_future.iloc[:2,:].copy(deep=True)
    # future_events_df2 = playoffs_future.iloc[2:4,:].copy(deep=True)
    future_events_df3 = playoffs_future.iloc[4:6, :].copy(deep=True)
    future_events_df4 = playoffs_future.iloc[6:8, :].copy(deep=True)

    def global_modeling():  ### GLOBAL MODELLING - NO EXOGENOUS VARIABLES
        log.debug("Global Modeling - No exogenous variables - Split df")
        m = NeuralProphet(n_forecasts=2, n_lags=10, epochs=EPOCHS, batch_size=BATCH_SIZE)
        train_input, test_input = m.split_df([df1_0, df2_0], freq="D")
        log.info("List Train size: {}".format(len(train_input)))
        log.info("Dfs in the list Train size: df0 = {}, df1= {}".format(len(train_input[0]), len(train_input[1])))
        log.info("Final Train dates: {}".format(train_input[1]["ds"][-2:]))
        log.info("List Test size: {}".format(len(test_input)))
        log.info("Initial Test dates: {}".format(test_input["ds"][:2]))
        metrics = m.fit(train_input, freq="D")
        forecast = m.predict(df=test_input)
        if PLOT:
            forecast = forecast if isinstance(forecast, list) else [forecast]
            for frst in forecast:
                fig1 = m.plot(frst)
                fig2 = m.plot_components(frst)
        log.debug("Global Modeling - No exogenous variables")
        train_input = {0: df1_0, 1: [df1_0, df2_0], 2: [df1_0, df2_0]}
        test_input = {0: df3_0, 1: df3_0, 2: [df3_0, df4_0]}
        info_input = {
            0: "Testing df train / df test - no events, no regressors",
            1: "Testing LIST df train / df test - no events, no regressors",
            2: "Testing LIST df train / LIST df test - no events, no regressors",
        }
        for i in range(0, 3):
            log.debug(info_input[i])
            m = NeuralProphet(n_forecasts=2, n_lags=10, epochs=EPOCHS, batch_size=BATCH_SIZE)
            metrics = m.fit(train_input[i], freq="D")
            forecast = m.predict(df=test_input[i])
            if PLOT:
                forecast = forecast if isinstance(forecast, list) else [forecast]
                for frst in forecast:
                    fig1 = m.plot(frst)
                    fig2 = m.plot_components(frst)

    def global_modeling_regressors():  ### GLOBAL MODELLING + REGRESSORS
        log.debug("Global Modeling + Regressors")
        train_input = {0: df1, 1: [df1, df2], 2: [df1, df2], 3: [df1, df2]}
        test_input = {0: df3, 1: df3, 2: [df3, df4], 3: [df3, df4]}
        regressors_input = {
            0: future_regressors_df3,
            1: future_regressors_df3,
            2: future_regressors_df3,
            3: [future_regressors_df3, future_regressors_df4],
        }
        info_input = {
            0: "Testing df train / df test - df regressor, no events",
            1: "Testing LIST df train / df test - df regressors, no events",
            2: "Testing LIST df train / LIST df test - df regressors, no events",
            3: "Testing LIST df train / LIST df test - LIST regressors, no events",
        }
        for i in range(0, 4):
            log.debug(info_input[i])
            m = NeuralProphet(n_forecasts=2, n_lags=10, epochs=EPOCHS, batch_size=BATCH_SIZE)
            m = m.add_lagged_regressor(names="A")
            metrics = m.fit(train_input[i], freq="D")
            future = m.make_future_dataframe(
                test_input[i], n_historic_predictions=True, regressors_df=regressors_input[i]
            )
            forecast = m.predict(df=future)
        if PLOT:
            forecast = forecast if isinstance(forecast, list) else [forecast]
            for frst in forecast:
                fig = m.PLOT(frst)
                fig = m.plot_components(frst)

    def global_modeling_events():  ### GLOBAL MODELLING + EVENTS
        log.debug("Global Modeling + Events")
        events_input = {
            0: future_events_df3,
            1: future_events_df3,
            2: future_events_df3,
            3: [future_events_df3, future_events_df4],
        }
        info_input = {
            0: "Testing df train / df test - df events, no regressors",
            1: "Testing LIST train / df test - df events, no regressors",
            2: "Testing LIST train / LIST test - df events, no regressors",
            3: "Testing LIST train / LIST test - LIST events, no regressors",
        }
        for i in range(0, 4):
            log.debug(info_input[i])
            m = NeuralProphet(n_forecasts=2, n_lags=10, epochs=EPOCHS, batch_size=BATCH_SIZE)
            m.add_events(["playoff"])
            history_df1 = m.create_df_with_events(df1_0, history_events_df1)
            history_df3 = m.create_df_with_events(df3_0, history_events_df3)
            if i > 0:
                history_df2 = m.create_df_with_events(df2_0, history_events_df2)
                history_df1 = [history_df1, history_df2]
            if i > 1:
                history_df4 = m.create_df_with_events(df4_0, history_events_df4)
                history_df3 = [history_df3, history_df4]
            metrics = m.fit(history_df1, freq="D")
            future = m.make_future_dataframe(history_df3, n_historic_predictions=True, events_df=events_input[i])
            forecast = m.predict(future)
            forecast = m.predict(df=future)
        if PLOT:
            forecast = forecast if isinstance(forecast, list) else [forecast]
            for frst in forecast:
                fig = m.plot(frst)
                fig = m.plot_components(frst)

    def global_modeling_events_plus_regressors():  ### GLOBAL MODELLING + REGRESSORS + EVENTS
        # One must call global modeling events and global modeling regressors before calling this function
        log.debug("Global Modeling + Events + Regressors")
        m = NeuralProphet(n_lags=10, n_forecasts=5, epochs=EPOCHS, batch_size=BATCH_SIZE)
        m = m.add_events(["playoff"])
        m = m.add_lagged_regressor(names="A")
        history_df1 = m.create_df_with_events(df1, history_events_df1)
        history_df2 = m.create_df_with_events(df2, history_events_df2)
        history_df3 = m.create_df_with_events(df3, history_events_df3)
        history_df4 = m.create_df_with_events(df4, history_events_df4)
        metrics = m.fit([history_df1, history_df2], freq="D")
        future = m.make_future_dataframe(
            [history_df3, history_df4],
            n_historic_predictions=True,
            events_df=[future_events_df3, future_events_df4],
            regressors_df=[future_regressors_df3, future_regressors_df4],
        )
        forecast = m.predict(future)
        if PLOT:
            for frst in forecast:
                fig = m.PLOT(frst)
                fig = m.plot_components(frst)

    global_modeling()
    global_modeling_regressors()
    global_modeling_events()
    global_modeling_events_plus_regressors()
    log.debug("GLOBAL MODELING TESTING - DONE")


def test_minimal():
    log.info("testing: Plotting")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=14,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    metrics_df = m.fit(df, freq="D", minimal=True)
    assert metrics_df is None
    forecast = m.predict(df)


def test_metrics():
    log.info("testing: Plotting")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        collect_metrics=["MAE", "MSE", "RMSE"],
    )
    metrics_df = m.fit(df, freq="D")
    assert metrics_df is not None
    forecast = m.predict(df)
