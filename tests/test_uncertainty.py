#!/usr/bin/env python3

import logging
import os
import pathlib

import pandas as pd
import pytest

from neuralprophet import NeuralProphet, uncertainty_evaluate

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


def test_uncertainty_estimation_peyton_manning():
    log.info("testing: Uncertainty Estimation Peyton Manning")
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
        n_forecasts=1,
        loss_func="Huber",
        quantiles=[0.01, 0.99],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )

    # add lagged regressors
    if m.n_lags > 0:
        df["A"] = df["y"].rolling(7, min_periods=1).mean()
        df["B"] = df["y"].rolling(30, min_periods=1).mean()
        m = m.add_lagged_regressor(name="A")
        m = m.add_lagged_regressor(name="B", only_last_value=True)

    # add events
    m = m.add_events(["superbowl", "playoff"], lower_window=-1, upper_window=1, regularization=0.1)

    m = m.add_country_holidays("US", mode="additive", regularization=0.1)

    df["C"] = df["y"].rolling(7, min_periods=1).mean()
    df["D"] = df["y"].rolling(30, min_periods=1).mean()

    m = m.add_future_regressor(name="C", regularization=0.1)
    m = m.add_future_regressor(name="D", regularization=0.1)

    history_df = m.create_df_with_events(df, events_df)

    m.fit(history_df, freq="D")

    periods = 90
    regressors_future_df = pd.DataFrame(data={"C": df["C"][:periods], "D": df["D"][:periods]})
    future_df = m.make_future_dataframe(
        df=history_df,
        regressors_df=regressors_future_df,
        events_df=events_df,
        periods=periods,
        n_historic_predictions=360,
    )
    forecast = m.predict(df=future_df)


def test_uncertainty_estimation_yosemite_temps():
    log.info("testing: Uncertainty Estimation Yosemite Temps")
    df = pd.read_csv(YOS_FILE)
    m = NeuralProphet(
        n_lags=12,
        n_forecasts=6,
        quantiles=[0.01, 0.99],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )

    metrics_df = m.fit(df, freq="5min")
    future = m.make_future_dataframe(df, periods=6, n_historic_predictions=3 * 24 * 12)
    forecast = m.predict(future)
    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)


def test_uncertainty_estimation_air_travel():
    log.info("testing: Uncertainty Estimation Air Travel")
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet(
        seasonality_mode="multiplicative",
        loss_func="MSE",
        quantiles=[0.01, 0.99],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    metrics_df = m.fit(df, freq="MS")
    future = m.make_future_dataframe(df, periods=50, n_historic_predictions=len(df))
    forecast = m.predict(future)


def test_uncertainty_estimation_multiple_quantiles():
    log.info("testing: Uncertainty Estimation Air Travel")
    df = pd.read_csv(AIR_FILE)
    multi_quantiles = [
        [0.5],  # forecast shows only yhat1, no duplicate yhat1 50.0%
        [0.8],  # forecast yhat1 and yhat1 80.0%
        [0.3, 0.6, 0.9],
        [0.05, 0.25, 0.75, 0.95],
    ]
    for quantiles in multi_quantiles:
        m = NeuralProphet(
            seasonality_mode="multiplicative",
            loss_func="MSE",
            quantiles=quantiles,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
        )
        metrics_df = m.fit(df, freq="MS")
        future = m.make_future_dataframe(df, periods=50, n_historic_predictions=len(df))
        forecast = m.predict(future)


def test_split_conformal_prediction():
    log.info("testing: Naive Split Conformal Prediction Air Travel")
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet(
        seasonality_mode="multiplicative",
        loss_func="MSE",
        quantiles=[0.05, 0.95],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )

    train_df, test_df = m.split_df(df, freq="MS", valid_p=0.2)
    train_df, cal_df = m.split_df(train_df, freq="MS", valid_p=0.15)
    metrics_df = m.fit(train_df, freq="MS")

    alpha = 0.1
    decompose = False
    for method in ["naive", "cqr"]:  # Naive and CQR SCP methods
        future = m.make_future_dataframe(
            test_df,
            periods=50,
            n_historic_predictions=len(test_df),
        )
        forecast = m.conformal_predict(
            future,
            calibration_df=cal_df,
            alpha=alpha,
            method=method,
            decompose=decompose,
        )
        eval_df = uncertainty_evaluate(forecast)


def test_asymmetrical_quantiles():
    log.info(
        "testing: Naive Split Conformal Prediction and Conformalized Quantile Regression " "with asymmetrical quantiles"
    )
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet(
        seasonality_mode="multiplicative",
        loss_func="MSE",
        quantiles=[0.05, 0.95],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )

    train_df, test_df = m.split_df(df, freq="MS", valid_p=0.2)
    train_df, cal_df = m.split_df(train_df, freq="MS", valid_p=0.15)
    metrics_df = m.fit(train_df, freq="MS")

    alpha = (0.03, 0.07)
    decompose = False
    future = m.make_future_dataframe(
        test_df,
        periods=50,
        n_historic_predictions=len(test_df),
    )

    # should raise value error if method is naive and alpha is not a float
    method = "naive"
    with pytest.raises(ValueError):
        forecast = m.conformal_predict(
            future,
            calibration_df=cal_df,
            alpha=alpha,
            method=method,
            decompose=decompose,
        )

    # should not raise value error if method is cqr and alpha is not a float
    method = "cqr"
    forecast = m.conformal_predict(
        future,
        calibration_df=cal_df,
        alpha=alpha,
        method=method,
        decompose=decompose,
    )
    eval_df = uncertainty_evaluate(forecast)
