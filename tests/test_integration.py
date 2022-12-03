#!/usr/bin/env python3

import logging
import math
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import torchmetrics

from neuralprophet import NeuralProphet, df_utils, forecaster, set_random_seed

log = logging.getLogger("NP.test")
log.setLevel("DEBUG")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 256
EPOCHS = 1
BATCH_SIZE = 128
LR = 1.0

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
        ar_reg=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    df = pd.read_csv(PEYTON_FILE, nrows=95)
    df, _ = df_utils.check_dataframe(df, check_y=False)
    df = m._handle_missing_data(df, freq="D", predicting=False)
    df_train, df_test = m.split_df(df, freq="D", valid_p=0.1)
    metrics = m.fit(df_train, freq="D", validation_df=df_test)
    val_metrics = m.test(df_test)
    log.debug("Metrics: train/eval: \n {}".format(metrics.to_string(float_format=lambda x: "{:6.3f}".format(x))))
    log.debug("Metrics: test: \n {}".format(val_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x))))


def test_df_utils_func():
    log.info("testing: df_utils Test")
    df = pd.read_csv(PEYTON_FILE, nrows=95)
    df, _ = df_utils.check_dataframe(df, check_y=False)

    # test find_time_threshold
    df, _, _, _ = df_utils.prep_or_copy_df(df)
    time_threshold = df_utils.find_time_threshold(df, n_lags=2, n_forecasts=2, valid_p=0.2, inputs_overbleed=True)
    df_train, df_val = df_utils.split_considering_timestamp(
        df, n_lags=2, n_forecasts=2, inputs_overbleed=True, threshold_time_stamp=time_threshold
    )

    # test find_time_threshold
    time_interval = df_utils.find_valid_time_interval_for_cv(df)

    # test unfold fold of dicts
    df1 = df.copy(deep=True)
    df1["ID"] = "df1"
    df2 = df.copy(deep=True)
    df2["ID"] = "df2"
    df_global = pd.concat((df1, df2))
    folds_dict = {}
    start_date, end_date = df_utils.find_valid_time_interval_for_cv(df_global)
    for df_name, df_i in df_global.groupby("ID"):
        # Use data only from the time period of intersection among time series
        mask = (df_i["ds"] >= start_date) & (df_i["ds"] <= end_date)
        df_i = df_i[mask].copy(deep=True)
        folds_dict[df_name] = df_utils._crossvalidation_split_df(
            df_i, n_lags=5, n_forecasts=2, k=5, fold_pct=0.1, fold_overlap_pct=0
        )
    folds = df_utils.unfold_dict_of_folds(folds_dict, 5)
    with pytest.raises(AssertionError):
        folds = df_utils.unfold_dict_of_folds(folds_dict, 3)

    # init data params with a list
    global_data_params = df_utils.init_data_params(df_global, normalize="soft")
    global_data_params = df_utils.init_data_params(df_global, normalize="soft1")
    global_data_params = df_utils.init_data_params(df_global, normalize="standardize")

    log.debug(f"Time Threshold: \n {time_threshold}")
    log.debug(f"Df_train: \n {df_train}")
    log.debug(f"Df_val: \n {df_val}")


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
        learning_rate=LR,
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
    log.debug(f"dates: {dates}")
    log.debug(f"dates_list: {dates_list}")
    log.debug(f"dates_array: {dates_array.dtype} {dates_array}")
    for cp in [dates_list, dates_array]:
        m = NeuralProphet(
            changepoints=cp,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
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
        learning_rate=LR,
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
    m = NeuralProphet(
        yearly_seasonality=8,
        weekly_seasonality=4,
        seasonality_mode="additive",
        seasonality_reg=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)
    log.debug("SUM of yearly season params: {}".format(sum(abs(m.model.season_params["yearly"].data.numpy()))))
    log.debug("SUM of weekly season params: {}".format(sum(abs(m.model.season_params["weekly"].data.numpy()))))
    log.debug(f"season params: {m.model.season_params.items()}")
    if PLOT:
        m.plot(forecast)
        # m.plot_components(forecast)
        m.plot_parameters()
        plt.show()
    log.info("testing: Seasonality: multiplicative")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True,
    #                   epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,)
    m = NeuralProphet(
        yearly_seasonality=8,
        weekly_seasonality=4,
        seasonality_mode="multiplicative",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)


def test_custom_seasons():
    log.info("testing: Custom Seasonality")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
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
        learning_rate=LR,
    )
    m = m.add_seasonality(name="quarterly", period=90, fourier_order=5)
    log.debug(f"seasonalities: {m.config_season.periods}")
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=365, periods=365)
    forecast = m.predict(df=future)
    log.debug(f"season params: {m.model.season_params.items()}")
    if PLOT:
        m.plot(forecast)
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
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=90)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot_latest_forecast(forecast, include_previous_forecasts=3)
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
        ar_reg=0.5,
        yearly_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=90)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot_latest_forecast(forecast, include_previous_forecasts=3)
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
        learning_rate=LR,
    )
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=90)
    forecast = m.predict(df=future)
    if PLOT:
        m.plot_latest_forecast(forecast, include_previous_forecasts=3)
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
        learning_rate=LR,
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    df["B"] = df["y"].rolling(30, min_periods=1).mean()
    m = m.add_lagged_regressor(names="A")
    m = m.add_lagged_regressor(names="B")
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, n_historic_predictions=10)
    forecast = m.predict(future)
    if PLOT:
        print(forecast.to_string())
        m.plot_latest_forecast(forecast, include_previous_forecasts=5)
        m.plot(forecast)
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
        learning_rate=LR,
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
        learning_rate=LR,
    )
    # set event windows
    m = m.add_events(
        ["superbowl", "playoff"], lower_window=-1, upper_window=1, mode="multiplicative", regularization=0.5
    )
    # add the country specific holidays
    m = m.add_country_holidays("US", mode="additive", regularization=0.5)
    m.add_country_holidays("Indonesia")
    m.add_country_holidays("Thailand")
    m.add_country_holidays("Philippines")
    m.add_country_holidays("Pakistan")
    m.add_country_holidays("Belarus")
    history_df = m.create_df_with_events(df, events_df)
    metrics_df = m.fit(history_df, freq="D")
    future = m.make_future_dataframe(df=history_df, events_df=events_df, periods=30, n_historic_predictions=90)
    forecast = m.predict(df=future)
    log.debug(f"Event Parameters:: {m.model.event_params}")
    if PLOT:
        m.plot_components(forecast)
        m.plot(forecast)
        m.plot_parameters()
        plt.show()


def test_future_reg():
    log.info("testing: Future Regressors")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS + 50)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
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
        m.plot(forecast)
        m.plot_components(forecast)
        m.plot_parameters()
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
        learning_rate=LR,
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
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)
    checksum1 = sum(forecast["yhat1"].values)
    set_random_seed(0)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)
    checksum2 = sum(forecast["yhat1"].values)
    set_random_seed(1)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)
    checksum3 = sum(forecast["yhat1"].values)
    log.debug(f"should be same: {checksum1} and {checksum2}")
    log.debug(f"should not be same: {checksum1} and {checksum3}")
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
        learning_rate=LR,
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
        m = NeuralProphet(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
        )
        folds = m.crossvalidation_split_df(df, freq="D", k=5, fold_pct=0.1, fold_overlap_pct=0.5)
        assert all([70 + i * 5 == len(train) for i, (train, val) in enumerate(folds)])
        assert all([10 == len(val) for (train, val) in folds])

    def check_cv(df, freq, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct):
        m = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
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
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        loss_func="MSE",
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)


def test_loss_func_torch():
    log.info("TEST setting torch.nn loss func")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        loss_func=torch.nn.MSELoss,
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=10, n_historic_predictions=10)
    forecast = m.predict(future)


def test_callable_loss():
    log.info("TEST Callable Loss")

    def my_loss(output, target):
        assym_penalty = 1.25
        beta = 1
        e = target - output
        me = torch.abs(e)
        z = torch.where(me < beta, 0.5 * (me**2) / beta, me - 0.5 * beta)
        z = torch.where(e < 0, z, assym_penalty * z)
        return z

    df = pd.read_csv(YOS_FILE, nrows=NROWS)
    # auto-lr with range test
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seasonality_mode="multiplicative",
        loss_func=my_loss,
    )
    metrics = m.fit(df, freq="5min")
    future = m.make_future_dataframe(df, periods=12 * 24, n_historic_predictions=12 * 24)
    forecast = m.predict(future)


def test_custom_torch_loss():
    log.info("TEST PyTorch Custom Loss")

    class MyLoss(torch.nn.modules.loss._Loss):
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

    df = pd.read_csv(YOS_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        loss_func=MyLoss,
    )
    metrics = m.fit(df, freq="5min")
    future = m.make_future_dataframe(df, periods=12, n_historic_predictions=12)
    forecast = m.predict(future)


def test_global_modeling_split_df():
    ### GLOBAL MODELLING - SPLIT DF
    log.info("Global Modeling - Split df")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1 = df.iloc[:128, :].copy(deep=True)
    df1["ID"] = "dataset1"
    df2 = df.iloc[128:256, :].copy(deep=True)
    df2["ID"] = "dataset2"
    df3 = df.iloc[256:384, :].copy(deep=True)
    df3["ID"] = "dataset3"
    df_global = pd.concat((df1, df2, df3))
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=2,
        n_lags=3,
    )
    log.info("split df with single ts df")
    df_train, df_val = m.split_df(df1)
    log.info("split df with many ts df")
    df_train, df_val = m.split_df(df_global)
    log.info("split df with many ts df - local_split")
    df_train, df_val = m.split_df(df_global, local_split=True)


def test_global_modeling_no_exogenous_variable():
    ### GLOBAL MODELLING - NO EXOGENOUS VARIABLE
    log.info("Global Modeling - No exogenous variables")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1_0 = df.iloc[:128, :].copy(deep=True)
    df1_0["ID"] = "df1"
    df2_0 = df.iloc[128:256, :].copy(deep=True)
    df2_0["ID"] = "df2"
    df3_0 = df.iloc[256:384, :].copy(deep=True)
    df3_0["ID"] = "df1"
    df4_0 = df.iloc[384:, :].copy(deep=True)
    df4_0["ID"] = "df2"
    train_input = {0: df1_0, 1: pd.concat((df1_0, df2_0)), 2: pd.concat((df1_0, df2_0))}
    test_input = {0: df3_0, 1: df3_0, 2: pd.concat((df3_0, df4_0))}
    info_input = {
        0: "Testing single ts df train / df test - no events, no regressors",
        1: "Testing many ts df train / df test - no events, no regressors",
        2: "Testing many ts df train / many ts df test - no events, no regressors",
    }
    for i in range(0, 3):
        log.info(info_input[i])
        m = NeuralProphet(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            n_forecasts=2,
            n_lags=10,
            trend_global_local="global",
            season_global_local="global",
        )
        metrics = m.fit(train_input[i], freq="D")
        forecast = m.predict(df=test_input[i])
        forecast_trend = m.predict_trend(df=test_input[i])
        forecast_seasonal_componets = m.predict_seasonal_components(df=test_input[i])
        if PLOT:
            for key, df in forecast.groupby("ID"):
                fig1 = m.plot(df)
                fig2 = m.plot_parameters(df_name=key)
                fig3 = m.plot_parameters()
    df4_0["ID"] = "df4"
    with pytest.raises(ValueError):
        forecast = m.predict(df4_0)
    log.info("Error - df with id not provided in the train df (not in the data params ID)")
    with pytest.raises(ValueError):
        metrics = m.test(df4_0)
    log.info("Error - df with id not provided in the train df (not in the data params ID)")
    m = NeuralProphet(
        n_forecasts=2,
        n_lags=10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        trend_global_local="global",
        season_global_local="global",
    )
    m.fit(pd.concat((df1_0, df2_0)), freq="D")
    with pytest.raises(ValueError):
        forecast = m.predict(df4_0)
    log.info("unknown_data_normalization was not set to True")
    with pytest.raises(ValueError):
        metrics = m.test(df4_0)
    log.info("unknown_data_normalization was not set to True")
    with pytest.raises(ValueError):
        forecast_trend = m.predict_trend(df4_0)
    log.info("unknown_data_normalization was not set to True")
    with pytest.raises(ValueError):
        forecast_seasonal_componets = m.predict_seasonal_components(df4_0)
    log.info("unknown_data_normalization was not set to True")
    # Set unknown_data_normalization to True - now there should be no errors
    m.config_normalization.unknown_data_normalization = True
    forecast = m.predict(df4_0)
    metrics = m.test(df4_0)
    forecast_trend = m.predict_trend(df4_0)
    forecast_seasonal_componets = m.predict_seasonal_components(df4_0)
    m.plot_parameters(df_name="df1")
    m.plot_parameters()


def test_global_modeling_validation_df():
    log.info("Global Modeling + Local Normalization")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1_0 = df.iloc[:128, :].copy(deep=True)
    df1_0["ID"] = "df1"
    df2_0 = df.iloc[128:256, :].copy(deep=True)
    df2_0["ID"] = "df2"
    df3_0 = df.iloc[256:384, :].copy(deep=True)
    df_global = pd.concat((df1_0, df2_0))
    m = NeuralProphet(
        n_forecasts=2,
        n_lags=10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    with pytest.raises(ValueError):
        m.fit(df_global, freq="D", validation_df=df3_0)
    log.info("Error - name of validation df was not provided")
    m = NeuralProphet(
        n_forecasts=2,
        n_lags=10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m.fit(df_global, freq="D", validation_df=df2_0)
    # Now it works because we provide the name of the validation_df


def test_global_modeling_global_normalization():
    ### GLOBAL MODELLING - NO EXOGENOUS VARIABLES - GLOBAL NORMALIZATION
    log.info("Global Modeling + Global Normalization")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1_0 = df.copy(deep=True)
    df1_0["ID"] = "df1"
    df2_0 = df.copy(deep=True)
    df2_0["ID"] = "df2"
    df3_0 = df.copy(deep=True)
    df3_0["ID"] = "df3"
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=2,
        n_lags=10,
        global_normalization=True,
        trend_global_local="global",
        season_global_local="global",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    train_df = pd.concat((df1_0, df2_0))
    test_df = df3_0
    m.fit(train_df)
    future = m.make_future_dataframe(test_df)
    forecast = m.predict(future)
    metrics = m.test(test_df)
    forecast_trend = m.predict_trend(test_df)
    forecast_seasonal_componets = m.predict_seasonal_components(test_df)
    m.plot_parameters()
    m.plot_parameters(df_name="df1")


def test_global_modeling_with_future_regressors():
    ### GLOBAL MODELLING + REGRESSORS
    log.info("Global Modeling + Regressors")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1 = df.iloc[:128, :].copy(deep=True)
    df2 = df.iloc[128:256, :].copy(deep=True)
    df3 = df.iloc[256:384, :].copy(deep=True)
    df4 = df.iloc[384:, :].copy(deep=True)
    df1["A"] = df1["y"].rolling(30, min_periods=1).mean()
    df2["A"] = df2["y"].rolling(10, min_periods=1).mean()
    df3["A"] = df3["y"].rolling(40, min_periods=1).mean()
    df4["A"] = df4["y"].rolling(20, min_periods=1).mean()
    df1["ID"] = "df1"
    df2["ID"] = "df2"
    df3["ID"] = "df1"
    df4["ID"] = "df2"
    future_regressors_df3 = pd.DataFrame(data={"A": df3["A"].iloc[:30]})
    future_regressors_df3["ID"] = "df1"
    future_regressors_df4 = pd.DataFrame(data={"A": df4["A"].iloc[:40]})
    future_regressors_df4["ID"] = "df2"
    train_input = {0: df1, 1: pd.concat((df1, df2)), 2: pd.concat((df1, df2))}
    test_input = {0: df3, 1: df3, 2: pd.concat((df3, df4))}
    regressors_input = {
        0: future_regressors_df3,
        1: future_regressors_df3,
        2: pd.concat((future_regressors_df3, future_regressors_df4)),
    }
    info_input = {
        0: "Testing single ts df train / single ts df test - single regressor, no events",
        1: "Testing many ts df train / single ts df test - single regressor, no events",
        2: "Testing many ts df train / many ts df test - many regressors, no events",
    }
    for i in range(0, 3):
        log.info(info_input[i])
        m = NeuralProphet(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            trend_global_local="global",
            season_global_local="global",
        )
        m = m.add_future_regressor(name="A")
        metrics = m.fit(train_input[i], freq="D")
        future = m.make_future_dataframe(test_input[i], n_historic_predictions=True, regressors_df=regressors_input[i])
        forecast = m.predict(future)
        if PLOT:
            for key, df in forecast.groupby("ID"):
                fig1 = m.plot(df)
                fig2 = m.plot_parameters(df_name=key)
                fig3 = m.plot_parameters()
    # Possible errors with regressors
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m = m.add_future_regressor(name="A")
    metrics = m.fit(pd.concat((df1, df2)), freq="D")
    with pytest.raises(ValueError):
        future = m.make_future_dataframe(
            pd.concat((df3, df4)), n_historic_predictions=True, regressors_df=future_regressors_df3
        )
    log.info("Error - regressors df len is different than ts df len")
    future_regressors_df3["ID"] = "dfn"
    with pytest.raises(ValueError):
        future = m.make_future_dataframe(df3, n_historic_predictions=True, regressors_df=future_regressors_df3)
    log.info("Error - key for regressors not valid")


def test_global_modeling_with_lagged_regressors():
    ### GLOBAL MODELLING + REGRESSORS
    log.info("Global Modeling + Regressors")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1 = df.iloc[:128, :].copy(deep=True)
    df2 = df.iloc[128:256, :].copy(deep=True)
    df3 = df.iloc[256:384, :].copy(deep=True)
    df4 = df.iloc[384:, :].copy(deep=True)
    df1["A"] = df1["y"].rolling(30, min_periods=1).mean()
    df2["A"] = df2["y"].rolling(10, min_periods=1).mean()
    df3["A"] = df3["y"].rolling(40, min_periods=1).mean()
    df4["A"] = df4["y"].rolling(20, min_periods=1).mean()
    df1["ID"] = "df1"
    df2["ID"] = "df2"
    df3["ID"] = "df1"
    df4["ID"] = "df2"
    future_regressors_df3 = pd.DataFrame(data={"A": df3["A"].iloc[:30]})
    future_regressors_df4 = pd.DataFrame(data={"A": df4["A"].iloc[:40]})
    future_regressors_df3["ID"] = "df1"
    future_regressors_df4["ID"] = "df2"
    train_input = {0: df1, 1: pd.concat((df1, df2)), 2: pd.concat((df1, df2))}
    test_input = {0: df3, 1: df3, 2: pd.concat((df3, df4))}
    regressors_input = {
        0: future_regressors_df3,
        1: future_regressors_df3,
        2: pd.concat((future_regressors_df3, future_regressors_df4)),
    }
    info_input = {
        0: "Testing single ts df train / single ts df test - single df regressors, no events",
        1: "Testing many ts df train / many ts df test - single df regressors, no events",
        2: "Testing many ts df train / many ts df test - many df regressors, no events",
    }
    for i in range(0, 3):
        log.info(info_input[i])
        m = NeuralProphet(
            n_lags=5,
            n_forecasts=3,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            trend_global_local="global",
            season_global_local="global",
        )
        m = m.add_lagged_regressor(names="A")
        metrics = m.fit(train_input[i], freq="D")
        future = m.make_future_dataframe(test_input[i], n_historic_predictions=True, regressors_df=regressors_input[i])
        forecast = m.predict(future)
        if PLOT:
            for key, df in forecast.groupby("ID"):
                fig1 = m.plot(df)
                fig2 = m.plot_parameters(df_name=key)
                fig3 = m.plot_parameters()
    # Possible errors with regressors
    m = NeuralProphet(
        n_lags=5,
        n_forecasts=3,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m = m.add_lagged_regressor(names="A")
    metrics = m.fit(pd.concat((df1, df2)), freq="D")
    future = m.make_future_dataframe(
        pd.concat((df3, df4)), n_historic_predictions=True, regressors_df=future_regressors_df3
    )
    log.info("global model regressors with regressors df with not all IDs from original df")
    future_regressors_df3["ID"] = "dfn"
    with pytest.raises(ValueError):
        future = m.make_future_dataframe(df3, n_historic_predictions=True, regressors_df=future_regressors_df3)
    log.info("Error - key for regressors not valid")


def test_global_modeling_with_events_only():
    ### GLOBAL MODELLING + EVENTS
    log.info("Global Modeling + Events")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1_0 = df.iloc[:128, :].copy(deep=True)
    df2_0 = df.iloc[128:256, :].copy(deep=True)
    df3_0 = df.iloc[256:384, :].copy(deep=True)
    df4_0 = df.iloc[384:, :].copy(deep=True)
    df1_0["ID"] = "df1"
    df2_0["ID"] = "df2"
    df3_0["ID"] = "df1"
    df4_0["ID"] = "df2"
    playoffs_history = pd.DataFrame(
        {
            "event": "playoff",
            "ds": pd.to_datetime(
                [
                    "2007-12-13",
                    "2008-05-31",
                    "2008-06-04",
                    "2008-06-06",
                    "2008-06-09",
                    "2008-12-13",
                    "2008-12-25",
                    "2009-01-01",
                    "2009-01-15",
                    "2009-03-20",
                    "2009-04-20",
                    "2009-05-20",
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
                    "2008-06-10",
                    "2008-06-11",
                    "2008-12-15",
                    "2008-12-16",
                    "2009-01-26",
                    "2009-01-27",
                    "2009-06-05",
                    "2009-06-06",
                ]
            ),
        }
    )
    future_events_df3 = playoffs_future.iloc[4:6, :].copy(deep=True)
    future_events_df4 = playoffs_future.iloc[6:8, :].copy(deep=True)
    future_events_df3["ID"] = "df1"
    future_events_df4["ID"] = "df2"
    events_input = {
        0: future_events_df3,
        1: future_events_df3,
        2: pd.concat((future_events_df3, future_events_df4)),
    }

    info_input = {
        0: "Testing single ts df train / single ts df test - single df events, no regressors",
        1: "Testing many ts df train / single ts df test - single df events, no regressors",
        2: "Testing many ts df train / many ts df test - many df events, no regressors",
    }
    for i in range(0, 3):
        log.debug(info_input[i])
        m = NeuralProphet(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            trend_global_local="global",
            season_global_local="global",
        )
        m.add_events(["playoff"])
        history_df1 = m.create_df_with_events(df1_0, history_events_df1)
        history_df2 = m.create_df_with_events(df2_0, history_events_df2)
        history_df3 = m.create_df_with_events(df3_0, history_events_df3)
        history_df4 = m.create_df_with_events(df4_0, history_events_df4)
        if i == 1:
            history_df1 = pd.concat((history_df1, history_df2))
            history_df3 = history_df3
        if i == 2:
            history_df1 = pd.concat((history_df1, history_df2))
            history_df3 = pd.concat((history_df3, history_df4))
        metrics = m.fit(history_df1, freq="D")
        future = m.make_future_dataframe(history_df3, n_historic_predictions=True, events_df=events_input[i])
        forecast = m.predict(future)
        forecast = m.predict(df=future)
        if PLOT:
            for key, df in forecast.groupby("ID"):
                fig1 = m.plot(df)
                fig2 = m.plot_parameters(df_name=key)
                fig3 = m.plot_parameters()
    # Possible errors with events
    m = NeuralProphet(
        n_forecasts=2,
        n_lags=10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    m.add_events(["playoff"])
    metrics = m.fit(history_df1, freq="D")
    future = m.make_future_dataframe(history_df3, n_historic_predictions=True, events_df=future_events_df3)
    log.info("global model events with events df with not all IDs from original df")
    future_events_df3["ID"] = "dfn"
    with pytest.raises(ValueError):
        future = m.make_future_dataframe(
            history_df3, n_historic_predictions=True, events_df=pd.concat((future_events_df3, future_events_df4))
        )
    log.info("Error - key for events not valid")


def test_global_modeling_with_events_and_future_regressors():
    ### GLOBAL MODELLING + REGRESSORS + EVENTS
    log.info("Global Modeling + Events + Regressors")
    df = pd.read_csv(PEYTON_FILE, nrows=512)
    df1 = df.iloc[:128, :].copy(deep=True)
    df2 = df.iloc[128:256, :].copy(deep=True)
    df3 = df.iloc[256:384, :].copy(deep=True)
    df4 = df.iloc[384:, :].copy(deep=True)
    df1["A"] = df1["y"].rolling(30, min_periods=1).mean()
    df2["A"] = df2["y"].rolling(10, min_periods=1).mean()
    df3["A"] = df3["y"].rolling(40, min_periods=1).mean()
    df4["A"] = df4["y"].rolling(20, min_periods=1).mean()
    df1["ID"] = "df1"
    df2["ID"] = "df2"
    df3["ID"] = "df1"
    df4["ID"] = "df2"
    future_regressors_df3 = pd.DataFrame(data={"A": df3["A"].iloc[:30]})
    future_regressors_df4 = pd.DataFrame(data={"A": df4["A"].iloc[:40]})
    future_regressors_df3["ID"] = "df1"
    future_regressors_df4["ID"] = "df2"
    playoffs_history = pd.DataFrame(
        {
            "event": "playoff",
            "ds": pd.to_datetime(
                [
                    "2007-12-13",
                    "2008-05-31",
                    "2008-06-04",
                    "2008-06-06",
                    "2008-06-09",
                    "2008-12-13",
                    "2008-12-25",
                    "2009-01-01",
                    "2009-01-15",
                    "2009-03-20",
                    "2009-04-20",
                    "2009-05-20",
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
                    "2008-06-10",
                    "2008-06-11",
                    "2008-12-15",
                    "2008-12-16",
                    "2009-01-26",
                    "2009-01-27",
                    "2009-06-05",
                    "2009-06-06",
                ]
            ),
        }
    )
    future_events_df3 = playoffs_future.iloc[4:6, :].copy(deep=True)
    future_events_df4 = playoffs_future.iloc[6:8, :].copy(deep=True)
    future_events_df3["ID"] = "df1"
    future_events_df4["ID"] = "df2"
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        trend_global_local="global",
        season_global_local="global",
    )
    m = m.add_events(["playoff"])
    m = m.add_future_regressor(name="A")
    history_df1 = m.create_df_with_events(df1, history_events_df1)
    history_df2 = m.create_df_with_events(df2, history_events_df2)
    history_df3 = m.create_df_with_events(df3, history_events_df3)
    history_df4 = m.create_df_with_events(df4, history_events_df4)
    metrics = m.fit(pd.concat((history_df1, history_df2)), freq="D")
    future = m.make_future_dataframe(
        pd.concat((history_df3, history_df4)),
        n_historic_predictions=True,
        events_df=pd.concat((future_events_df3, future_events_df4)),
        regressors_df=pd.concat((future_regressors_df3, future_regressors_df4)),
    )
    forecast = m.predict(future)
    if PLOT:
        for key, df in forecast.groupby("ID"):
            fig1 = m.plot(df)
            fig2 = m.plot_parameters(df_name=key)
            fig3 = m.plot_parameters()


def test_auto_normalization():
    length = 100
    days = pd.date_range(start="2017-01-01", periods=length)
    y = np.ones(length)
    y[1] = 0
    y[2] = 2
    y[3] = 3.3
    df = pd.DataFrame({"ds": days, "y": y})
    df["future_constant"] = 1.0
    df["future_dynamic"] = df["y"] * 2
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=5,
        normalize="auto",
    )
    m = m.add_future_regressor("future_constant")
    m = m.add_future_regressor("future_dynamic")
    _ = m.fit(df, freq="D")


def test_minimal():
    log.info("testing: Plotting")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=7,
        n_lags=14,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="D", minimal=True)
    assert metrics_df is None
    forecast = m.predict(df)


def test_get_latest_forecast():
    log.info("testing: get_latest_forecast")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=24,
        n_lags=36,
        changepoints_range=0.95,
        n_changepoints=30,
        weekly_seasonality=False,
    )
    metrics_df = m.fit(df)
    forecast = m.predict(df)
    forecastdf = m.get_latest_forecast(forecast, df_name=None, include_history_data=None, include_previous_forecasts=5)
    forecastdf = m.get_latest_forecast(forecast, include_history_data=False, include_previous_forecasts=5)
    forecastdf = m.get_latest_forecast(forecast, include_history_data=True, include_previous_forecasts=5)
    help(m.get_latest_forecast)
    log.info("testing: get_latest_forecast with n_lags=0")
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=24,
        n_lags=0,
        changepoints_range=0.95,
        n_changepoints=30,
        weekly_seasonality=False,
    )
    metrics_df = m.fit(df)
    forecast = m.predict(df)
    with pytest.raises(Exception):
        m.get_latest_forecast(forecast, include_history_data=None, include_previous_forecasts=5)

    df1 = df.copy(deep=True)
    df1["ID"] = "df1"
    df2 = df.copy(deep=True)
    df2["ID"] = "df2"
    df_global = pd.concat((df1, df2))
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=24,
        n_lags=36,
        changepoints_range=0.95,
        n_changepoints=30,
        weekly_seasonality=False,
    )
    metrics_df = m.fit(df_global, freq="D")
    future = m.make_future_dataframe(df_global, periods=m.n_forecasts, n_historic_predictions=10)
    forecast = m.predict(future)
    log.info("Plot forecast with many IDs - Raise exceptions")
    forecast = m.predict(df_global)
    forecastdf = m.get_latest_forecast(forecast, df_name="df1", include_history_data=None, include_previous_forecasts=5)
    with pytest.raises(Exception):
        m.get_latest_forecast(forecast, include_previous_forecasts=10)


def test_metrics():
    log.info("testing: Plotting")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        collect_metrics=["MAE", "MSE", "RMSE"],
    )
    metrics_df = m.fit(df, freq="D")
    assert all([metric in metrics_df.columns for metric in ["MAE", "MSE", "RMSE"]])
    forecast = m.predict(df)

    m2 = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        collect_metrics={"ABC": torchmetrics.MeanAbsoluteError()},
    )
    metrics_df = m2.fit(df, freq="D")
    assert "ABC" in metrics_df.columns
    forecast = m2.predict(df)


def test_progress_display():
    log.info("testing: Progress Display")
    df = pd.read_csv(AIR_FILE, nrows=100)
    df_val = df[-20:]
    progress_types = ["bar", "print", "plot", "plot-all", "none"]
    for progress in progress_types:
        m = NeuralProphet(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
        )
        metrics_df = m.fit(df, progress=progress)


def test_n_lags_for_regressors():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    df1 = df.iloc[:128, :].copy(deep=True)
    df1["A"] = df1["y"].rolling(30, min_periods=1).mean()
    n_lags_input = [2, 2, 5, 2, 1, 2, 2, 0]
    n_lags_regressors_input = [2, "auto", 2, 5, 2, 1, "scalar", 5]
    info_input = [
        "n_lags == n_lags_regressors",
        "n_lags == n_lags_regressors (auto)",
        "n_lags > n_lags_regressors",
        "n_lags < n_lags_regressors",
        "n_lags (1) < n_lags_regressors",
        "n_lags > n_lags_regressors (1)",
        "n_lags > n_lags_regressors (scalar)",
        "n_lags == 0 and n_lags_regressors > 0",
    ]
    # Testing cases with 1 covariate
    for i in range(len(info_input)):
        log.debug(info_input[i])
        m = NeuralProphet(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            n_forecasts=2,
            n_lags=n_lags_input[i],
        )
        m = m.add_lagged_regressor(names="A", n_lags=n_lags_regressors_input[i])
        metrics = m.fit(df1, freq="D")
        future = m.make_future_dataframe(df1, n_historic_predictions=True)
        forecast = m.predict(df=future)
        if PLOT:
            fig = m.plot(forecast)
            fig = m.plot_parameters()
    # Testing case with 2 covariates
    df1["B"] = df1["y"].rolling(8, min_periods=1).mean()
    n_lags_input = [0, 2, 2, 5, 1]
    n_lags_regressors_input_A = [5, 7, 3, 3, "scalar"]
    n_lags_regressors_input_B = [7, 5, None, None, None]
    info_input = [
        "n_lags == 0 and 2 regressors with different lags between them",
        "n_lags > 0 and 2 regressors with different lags between them",
        "n_lags < lags from both regressors",
        "n_lags > lags from both regressors",
        "n_lags == lags from both regressors (scalar)",
    ]
    for i in range(len(info_input)):
        log.debug(info_input[i])
        m = NeuralProphet(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR, n_forecasts=3, n_lags=n_lags_input[i])
        if i < 2:
            m = m.add_lagged_regressor(names="A", n_lags=n_lags_regressors_input_A[i])
            m = m.add_lagged_regressor(names="B", n_lags=n_lags_regressors_input_B[i])
        else:
            # Testing call of add_lagged_regressor with list of names
            m = m.add_lagged_regressor(names=["A", "B"], n_lags=n_lags_regressors_input_A[i])
        metrics = m.fit(df1, freq="D")
        future = m.make_future_dataframe(df1, n_historic_predictions=True)
        forecast = m.predict(df=future)
        if PLOT:
            fig = m.plot(forecast)
            fig = m.plot_parameters()
    # Testing case with assertion error in time_dataset - n_lags = 0
    log.debug("Exception regressor n_lags == 0")
    m = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_forecasts=2,
        n_lags=2,
    )
    m = m.add_lagged_regressor(names="A", n_lags=0)
    m = m.add_lagged_regressor(names="B", n_lags=0)
    with pytest.raises(AssertionError):
        metrics = m.fit(df1, freq="D")


def test_drop_missing_values_after_imputation():
    m1 = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_lags=12,
        n_forecasts=12,
        weekly_seasonality=True,
        impute_missing=True,
        impute_linear=10,
        impute_rolling=0,
        drop_missing=True,
    )
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    log.info("introducing two large NaN windows")
    df.loc[100:131, "y"] = np.nan
    df.loc[170:200, "y"] = np.nan
    metrics = m1.fit(df, freq="D", validation_df=None)
    future = m1.make_future_dataframe(df, periods=60, n_historic_predictions=60)
    forecast = m1.predict(df=df)
    forecast = m1.predict(df=future)

    log.info("Testing drop of remaining values after lin imputation, no lags")
    m2 = NeuralProphet(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        n_lags=0,
        n_forecasts=12,
        weekly_seasonality=True,
        impute_missing=True,
        impute_linear=10,
        impute_rolling=0,
        drop_missing=True,
    )
    metrics = m2.fit(df, freq="D", validation_df=None)
    forecast = m2.predict(df=df)
    future = m2.make_future_dataframe(df, periods=60, n_historic_predictions=60)
    forecast = m2.predict(df=future)


def test_predict_raw():
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)

    # no quantiles
    m = NeuralProphet(n_forecasts=12, n_lags=24, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR)
    log.info("Testing raw prediction without any quantiles")
    metrics = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(df=future, raw=True)

    # with quantiles
    m = NeuralProphet(
        n_forecasts=12, n_lags=24, quantiles=[0.9, 0.1], epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR
    )
    log.info("Testing raw prediction with some quantiles")
    metrics = m.fit(df, freq="D")
    future = m.make_future_dataframe(df, periods=30, n_historic_predictions=100)
    forecast = m.predict(df=future, raw=True)


def test_accelerator():
    log.info("testing: accelerator in Lightning (if available)")
    df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    m = NeuralProphet(
        n_forecasts=2,
        n_lags=14,
        num_hidden_layers=2,
        d_hidden=32,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        trend_reg=0.1,
        quantiles=[0.1, 0.9],
        accelerator="auto",
    )
    df["A"] = df["y"].rolling(7, min_periods=1).mean()
    cols = [col for col in df.columns if col not in ["ds", "y"]]
    m = m.add_lagged_regressor(names=cols)
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)
    metrics_df = m.fit(df, freq="D")
    forecast = m.predict(df)
